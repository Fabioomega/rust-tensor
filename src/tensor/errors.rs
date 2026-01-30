#[derive(Debug)]
pub enum OpError<'a> {
    InvalidViewShape,
    NotSameShape(&'a [i32], &'a [i32]),
}

impl std::fmt::Display for OpError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpError::InvalidViewShape => write!(
                f,
                "the view shape does not have the same size as the original shape"
            ),
            OpError::NotSameShape(expected, got) => {
                write!(f, "expected {:?}, but got {:?}", *expected, *got)
            }
        }
    }
}

impl std::error::Error for OpError<'_> {}
