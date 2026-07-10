use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::slice;

pub struct FastMmapVec<T> {
    file: File,
    mmap: Option<MmapMut>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> FastMmapVec<T> {
    // create a new file
    pub fn new<P: AsRef<Path>>(path: P, length: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let mut slf = Self {
            file,
            mmap: None,
            len: 0,
            _marker: PhantomData,
        };
        slf.resize(length)?;
        Ok(slf)
    }

    /// change file size & make a new mmap
    pub fn resize(&mut self, new_len: usize) -> io::Result<()> {
        self.mmap = None;
        let type_size = mem::size_of::<T>() as u64;
        let new_byte_size = new_len as u64 * type_size;
        let current_byte_size = self.file.metadata()?.len();
        if new_byte_size < current_byte_size {
            self.file.set_len(new_byte_size)?;
        }
        else if new_byte_size > current_byte_size {
            let fd = self.file.as_raw_fd();
            let ret = rustix::fs::fallocate(
                unsafe { rustix::fd::BorrowedFd::borrow_raw(fd) },
                rustix::fs::FallocateFlags::empty(),
                0,
                new_byte_size,
            );
            if ret.is_err() {
                self.file.set_len(new_byte_size)?;
            }
        }
        if new_byte_size > 0 {
            let mmap = unsafe { MmapOptions::new().map_mut(&self.file)? };
            self.mmap = Some(mmap);
        }
        else {
            self.mmap = None;
        }
        self.len = new_len;
        Ok(())
    }

    // as_file pointer
    pub fn as_file(&self) -> &File {
        &self.file
    }
}


// slicing to u8, u32, u64, etc.
impl<T> Deref for FastMmapVec<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        match &self.mmap {
            Some(mmap) => unsafe {
                slice::from_raw_parts(mmap.as_ptr() as *const T, self.len)
            },
            None => &[],
        }
    }
}

// deref_mut
impl<T> DerefMut for FastMmapVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.mmap {
            Some(mmap) => unsafe {
                slice::from_raw_parts_mut(mmap.as_mut_ptr() as *mut T, self.len)
            },
            None => &mut [],
        }
    }
}