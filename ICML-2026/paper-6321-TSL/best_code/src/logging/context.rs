use std::cell::RefCell;

thread_local!(static EPOCH: RefCell<Option<usize>> = const { RefCell::new(None) });
thread_local!(static TREE_ID: RefCell<Option<usize>> = const { RefCell::new(None) });

pub fn with_epoch<F, R>(epoch: usize, f: F) -> R
where
    F: FnOnce() -> R,
{
    EPOCH.with(|e| {
        *e.borrow_mut() = Some(epoch);
        let result = f();
        *e.borrow_mut() = None;
        result
    })
}

pub fn with_tree_id<F, R>(tree_id: usize, f: F) -> R
where
    F: FnOnce() -> R,
{
    TREE_ID.with(|t| {
        *t.borrow_mut() = Some(tree_id);
        let result = f();
        *t.borrow_mut() = None;
        result
    })
}

pub fn current_epoch() -> Option<usize> {
    EPOCH.with(|e| *e.borrow())
}

pub fn current_tree_id() -> Option<usize> {
    TREE_ID.with(|t| *t.borrow())
}
