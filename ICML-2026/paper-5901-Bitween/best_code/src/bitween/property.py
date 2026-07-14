from sympy import Expr, sympify


class Property(Expr):
    """
    A property is an expression that is either true or false.
    """

    def __new__(cls, expr: Expr, time_spent=0, verified=None, **kwargs):
        """
        Create a new property.

        :param expr: the expression
        :param time_spent: time spent to generate the property
        :param verified: whether the property is verified or not
        :param kwargs: additional arguments
        """
        if not isinstance(expr, Expr):
            expr = sympify(expr)
        obj = Expr.__new__(cls, expr, **kwargs)
        obj._expr = expr
        obj._time_spent = time_spent
        obj._verified = verified
        obj._epsilon = None
        return obj

    @property
    def time_spent(self):
        """Get the time spent"""
        return self._time_spent

    @property
    def expr(self):
        """Get the expression"""
        return self._expr

    @property
    def verified(self):
        """Get the verified status"""
        return self._verified

    @verified.setter
    def verified(self, verified):
        """Set the verification status"""
        self._verified = verified

    @property
    def epsilon(self):
        """Get the epsilon value"""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        """Set the epsilon value"""
        self._epsilon = epsilon

    def __str__(self) -> str:
        return str(self._expr)

    def __repr__(self) -> str:
        return str(self._expr)

    def __eq__(self, other):
        if isinstance(other, Property):
            return self._expr == other._expr
        elif isinstance(other, Expr):
            return self._expr == other
        return False

    def __hash__(self):
        return hash(self._expr)

    def to_expr(self):
        return self._expr


class Equality(Property):
    """
    An equality property.
    """

    pass


class Inequality(Property):
    """
    An inequality property.
    """

    pass


if __name__ == "__main__":
    from sympy import symbols

    def test_create_property_from_expr():
        x, y, z = symbols("x y z")
        expr1 = x**2 + y**2
        prop1 = Property(expr1)
        assert prop1 == expr1
        assert str(prop1) == str(expr1)
        assert prop1.epsilon is None
        assert prop1.verified is None
        assert prop1.free_symbols == {x, y}

        expr2 = x + y + z
        prop2 = Property(expr2, verified=True)
        prop2.epsilon = 0.1
        assert str(prop2) == str(expr2)
        assert prop2 == expr2
        assert prop2.epsilon == 0.1
        assert prop2.verified is True
        assert prop2.free_symbols == {x, y, z}
        assert isinstance(Expr(Property(x + y)), Expr)
        assert isinstance(Property(x + y).to_expr(), Expr)

    test_create_property_from_expr()
