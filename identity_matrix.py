from sdk_registry import solver, method

@solver("IdentityMatrix")
class IdentityMatrix:
    """
    Standard Identity Matrix (I).
    I[i, j] = 1 if i == j, else 0.
    """
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)

    @method("IdentityMatrix", "get_element")
    def get_element(self, r: int, c: int) -> int:
        return 1 if r == c else 0
