# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools

import numpy as np
import scipy

from nums.core.storage.storage import ArrayGrid
from nums.core.array import utils as array_utils
from nums.core.systems.systems import System
from nums.core.array.blockarray import BlockArray
from nums.core.array.base import SparseBlock


class SparseBlockArray(BlockArray):

    def __init__(self, grid: ArrayGrid, system: System, blocks: np.ndarray = None):
        self.grid = grid
        self.system = system
        self.shape = self.grid.shape
        self.block_shape = self.grid.block_shape
        self.size = np.product(self.shape)
        self.ndim = len(self.shape)
        self.dtype = self.grid.dtype
        self.blocks = blocks
        if self.blocks is None:
            # TODO (hme): Subclass np.ndarray for self.blocks instances,
            #  and override key methods to better integrate with NumPy's ufuncs.
            self.blocks = np.empty(shape=self.grid.grid_shape, dtype=SparseBlock)
            for grid_entry in self.grid.get_entry_iterator():
                self.blocks[grid_entry] = SparseBlock(grid_entry=grid_entry,
                                                grid_shape=self.grid.grid_shape,
                                                rect=self.grid.get_slice_tuples(grid_entry),
                                                shape=self.grid.get_block_shape(grid_entry),
                                                dtype=self.dtype,
                                                transposed=False,
                                                system=self.system)
    def get(self) -> np.ndarray:
        result: np.ndarray = np.zeros(shape=self.grid.shape, dtype=self.grid.dtype)
        block_shape: np.ndarray = np.array(self.grid.block_shape, dtype=np.int)
        arrays: list = self.system.get([self.blocks[grid_entry].oid
                                        for grid_entry in self.grid.get_entry_iterator()])
        for block_index, grid_entry in enumerate(self.grid.get_entry_iterator()):
            start = block_shape * grid_entry
            entry_shape = np.array(self.grid.get_block_shape(grid_entry), dtype=np.int)
            end = start + entry_shape
            slices = tuple(map(lambda item: slice(*item), zip(*(start, end))))
            block = self.blocks[grid_entry]
            arr = arrays[block_index]
            if block.transposed:
                arr = arr.T
            
            result[slices] = arr.A

        return result

    def find(self, app) -> np.ndarray:
        rows, cols, vals = [], [], []
        block_shape: np.ndarray = np.array(self.grid.block_shape, dtype=np.int)
        arrays: list = self.system.get([self.blocks[grid_entry].oid
                                        for grid_entry in self.grid.get_entry_iterator()])
        for block_index, grid_entry in enumerate(self.grid.get_entry_iterator()):
            start = block_shape * grid_entry
            (r, c, v) = scipy.sparse.find(arrays[block_index])
            r = r + start[0]
            c = c + start[1]

            rows.append(r)
            cols.append(c)
            vals.append(v)

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)

        block_shape = app.get_block_shape(rows.shape, rows.dtype)
        val_block_shape = app.get_block_shape(vals.shape, vals.dtype)

        return (
            BlockArray.from_np(rows, block_shape, False, self.system),
            BlockArray.from_np(cols, block_shape, False, self.system),
            BlockArray.from_np(vals, val_block_shape, False, self.system),
            )


    @classmethod
    def from_np(cls, arr, block_shape, copy, system):
        dtype_str = str(arr.dtype)
        grid = ArrayGrid(arr.shape, block_shape, dtype_str)
        rarr = SparseBlockArray(grid, system)
        grid_entry_iterator = grid.get_entry_iterator()
        for grid_entry in grid_entry_iterator:
            grid_slice = grid.get_slice(grid_entry)
            block = scipy.sparse.csr_matrix(arr[grid_slice])

            rarr.blocks[grid_entry].oid = system.put(block)
            rarr.blocks[grid_entry].dtype = getattr(np, dtype_str)
        return rarr

    @classmethod
    def from_blocks(cls, arr: np.ndarray, result_shape, system):
        sample_idx = tuple(0 for dim in arr.shape)
        if isinstance(arr, SparseBlock):
            sample_block = arr
            result_shape = ()
        else:
            sample_block = arr[sample_idx]
            if result_shape is None:
                result_shape = array_utils.shape_from_block_array(arr)
        result_block_shape = sample_block.shape
        result_dtype_str = sample_block.dtype.__name__
        result_grid = ArrayGrid(shape=result_shape,
                                block_shape=result_block_shape,
                                dtype=result_dtype_str)
        assert arr.shape == result_grid.grid_shape
        result = SparseBlockArray(result_grid, system)
        for grid_entry in result_grid.get_entry_iterator():
            if isinstance(arr, SparseBlock):
                block: SparseBlock = arr
            else:
                block: SparseBlock = arr[grid_entry]
            result.blocks[grid_entry] = block
        return result

    def touch(self):
        oids = []
        for grid_entry in self.grid.get_entry_iterator():
            block = self.blocks[grid_entry]
            oids.append(self.system.touch(block.oid, syskwargs=block.syskwargs()))
        self.system.get(oids)
        return self

    def __matmul__(self, other):
        if len(self.shape) > 2:
            # TODO: (bcp) NumPy's implementation does a stacked matmul, which is not supported yet.
            raise NotImplementedError("Matrix multiply for tensors of rank > 2 not supported yet.")
        else:
            return self.tensordot(other, 1)

    def tensordot(self, other, axes=2):
        if not isinstance(other, SparseBlockArray):
            raise ValueError("Cannot automatically construct SparseBlockArray for tensor operations.")

        return self._tensordot(other, axes)

    def _tensordot(self, other, axes):
        this_axes = self.grid.grid_shape[:-axes]
        this_sum_axes = self.grid.grid_shape[-axes:]
        other_axes = other.grid.grid_shape[axes:]
        other_sum_axes = other.grid.grid_shape[:axes]
        assert this_sum_axes == other_sum_axes
        result_shape = tuple(self.shape[:-axes] + other.shape[axes:])
        result_block_shape = tuple(self.block_shape[:-axes] + other.block_shape[axes:])
        result_grid = ArrayGrid(shape=result_shape,
                                block_shape=result_block_shape,
                                dtype=array_utils.get_bop_output_type("tensordot",
                                                                      self.dtype,
                                                                      other.dtype).__name__)
        assert result_grid.grid_shape == tuple(this_axes + other_axes)
        result = SparseBlockArray(result_grid, self.system)
        this_dims = list(itertools.product(*map(range, this_axes)))
        other_dims = list(itertools.product(*map(range, other_axes)))
        sum_dims = list(itertools.product(*map(range, this_sum_axes)))
        for i in this_dims:
            for j in other_dims:
                grid_entry = tuple(i + j)
                result_block = None
                for k in sum_dims:
                    self_block = self.blocks[tuple(i + k)]
                    other_block = other.blocks[tuple(k + j)]
                    dotted_block = self_block.tensordot(other_block, axes=axes)
                    if result_block is None:
                        result_block = dotted_block
                    else:
                        result_block += dotted_block
                result.blocks[grid_entry] = result_block
        return result

    def csr(self):
        meta_swap = self.grid.to_meta()
        grid_swap = ArrayGrid.from_meta(meta_swap)
        rarr_src = np.ndarray(self.blocks.shape, dtype='O')

        for grid_entry in self.grid.get_entry_iterator():
            rarr_src[grid_entry] = self.blocks[grid_entry].csr()
        
        rarr_swap = SparseBlockArray(grid_swap, self.system, rarr_src)
        return rarr_swap

    def csc(self):
        meta_swap = self.grid.to_meta()
        grid_swap = ArrayGrid.from_meta(meta_swap)
        rarr_src = np.ndarray(self.blocks.shape, dtype='O')

        for grid_entry in self.grid.get_entry_iterator():
            rarr_src[grid_entry] = self.blocks[grid_entry].csc()
        
        rarr_swap = SparseBlockArray(grid_swap, self.system, rarr_src)
        return rarr_swap
       