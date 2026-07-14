from sympy import Function, simplify

from bitween.property import Property


def make_set(x):
    return {x}


def find_set(x, sets: list[set]):
    for s in sets:
        if x in s:
            return s
    return None


def create_equivalence_classes(properties: list[Property]) -> list[set[Property]]:
    """
    This method creates equivalence classes of properties. The property needs to be
    in to form of a Property object having an expr attribute which equals to zero.
    """
    sets = [make_set(x) for x in properties]
    for i in range(len(properties)):
        for j in range(i + 1, len(properties)):
            if simplify(properties[i].expr - properties[j].expr) == 0:
                s1 = find_set(properties[i], sets)
                s2 = find_set(properties[j], sets)
                if s1 != s2:
                    sets.remove(s1)
                    sets.remove(s2)
                    sets.append(s1.union(s2))
    return sets


def get_smallest_properties(classes: list[set[Property]]) -> set[Property]:
    """Get the smallest expression in a set of expressions"""
    sets = set()
    for cls in classes:
        smallest = None
        for e in cls:
            if smallest is None or e.count_ops() < smallest.count_ops():
                smallest = e
        sets.add(smallest)
    return sets


def refine(properties: list[Property]):
    """
    Refine the properties by removing the redundant ones.

    Remarks
    -------
    This method creates equivalence classes of properties. It doesn't interprets
    function symbols.
    """
    return get_smallest_properties(create_equivalence_classes(properties))


# Test Helper
def check_sets_equal(list1: list[set[Property]], list2: list[set[Property]]) -> bool:
    """Check if two lists of sets are equal"""
    set1 = set(frozenset(s) for s in list1)
    set2 = set(frozenset(s) for s in list2)
    return set1 == set2
    # if len(list1) != len(list2):
    #     return False
    # for set1 in list1:
    #     found_match = False
    #     for set2 in list2:
    #         if set1 == set2:
    #             found_match = True
    #             break
    #     if not found_match:
    #         return False
    # return True


# Test
def test_create_equivalence_classes():
    x = Symbol("x", real=True)
    f = Function("f")

    properties = [
        Property((f(x) ** 3 + f(x) ** 2 - f(x) - 1) / (f(x) ** 2 + 2 * f(x) + 1)),
        Property(f(x) - 1),
        Property((f(x) + 1) * (f(x) - 2) - (f(x) - 1) * f(x)),
        Property(-2),
        Property(f(x) + 3),
    ]
    for property in properties:
        print(f"{property.count_ops()} -- {property}")

    expected = [
        {
            Property((f(x) ** 3 + f(x) ** 2 - f(x) - 1) / (f(x) ** 2 + 2 * f(x) + 1)),
            Property(f(x) - 1),
        },
        {Property((f(x) + 1) * (f(x) - 2) - (f(x) - 1) * f(x)), Property(-2)},
        {Property(f(x) + 3)},
    ]
    smallest = {Property(-2), Property(f(x) + 3), Property(f(x) - 1)}
    eq_classes = create_equivalence_classes(properties)
    assert check_sets_equal(eq_classes, expected)
    found = get_smallest_properties(eq_classes)
    print(f"smallest: {smallest}")
    print(f"found   : {found}")
    assert smallest == found

    properties = [
        Property(f(x) ** 2 - 1),
        Property((f(x) - 1) * (f(x) + 1)),
        Property(f(x) ** 2 - 4),
        Property(f(x) ** 2 - 2 * f(x) + 1),
    ]
    expected = [
        {Property(f(x) ** 2 - 2 * f(x) + 1)},
        {Property(f(x) ** 2 - 1), Property((f(x) - 1) * (f(x) + 1))},
        {Property(f(x) ** 2 - 4)},
    ]
    smallest = {
        Property(f(x) ** 2 - 1),
        Property(f(x) ** 2 - 4),
        Property(f(x) ** 2 - 2 * f(x) + 1),
    }
    eq_classes = create_equivalence_classes(properties)
    assert check_sets_equal(eq_classes, expected)
    found = get_smallest_properties(eq_classes)
    print(f"smallest: {smallest}")
    print(f"found   : {found}")
    assert smallest == found


def test_squared_function():
    from sympy import Function, symbols

    x, y = symbols("x y")
    f = Function("f")
    # fmt: off
    properties = [
        2*f(x) + 2*f(y) - f(x - y) - f(x + y),
        2*f(x)*f(y) + 2*f(x)*f(x - y) + 2*f(x)*f(x + y) - 2*f(x) + 2*f(y)**2 + f(y)*f(x - y) + f(y)*f(x + y) - 2*f(y) - f(x - y)**2 - 2*f(x - y)*f(x + y) + f(x - y) - f(x + y)**2 + f(x + y),
        f(x)**2 - 2*f(x)*f(y) - 2*f(x)*f(x + y) + 2*f(x) + f(y)**2 - 2*f(y)*f(x + y) + 2*f(y) - f(x - y) + f(x + y)**2 - f(x + y),
        2*f(x)*f(y) + 2*f(x)*f(x - y) - 2*f(x) + 2*f(y)**2 + f(y)*f(x - y) - f(y)*f(x + y) - 2*f(y) - f(x - y)**2 - f(x - y)*f(x + y) + f(x - y) + f(x + y),
        2*f(x)**2 + 2*f(x)*f(y) + f(x)*f(x - y) + f(x)*f(x + y) - 2*f(x) + 2*f(y)*f(x - y) + 2*f(y)*f(x + y) - 2*f(y) - f(x - y)**2 - 2*f(x - y)*f(x + y) + f(x - y) - f(x + y)**2 + f(x + y),
        2*f(x)*f(x - y) - 2*f(x)*f(x + y) + 2*f(x) + 2*f(y)*f(x - y) - 2*f(y)*f(x + y) + 2*f(y) - f(x - y)**2 - f(x - y) + f(x + y)**2 - f(x + y),
        f(x)**2 - 2*f(x)*f(y) - 2*f(x)*f(x - y) + 2*f(x)*f(x + y) - 2*f(x) + f(y)**2 - 2*f(y)*f(x - y) + 2*f(y)*f(x + y) - 2*f(y) + f(x - y)**2 - f(x - y)*f(x + y) + f(x - y) - f(x + y)**2 + f(x + y),
        f(x)**2 - 2*f(x)*f(y) - 2*f(x) + f(y)**2 - 2*f(y) - f(x - y)*f(x + y) + f(x - y) + f(x + y),
        f(x)**2 - 2*f(x)*f(y) - 2*f(x)*f(x - y) + 2*f(x)*f(x + y) + 2*f(x) + f(y)**2 - 2*f(y)*f(x - y) + 2*f(y)*f(x + y) + 2*f(y) + f(x - y)**2 - f(x - y)*f(x + y) - f(x - y) - f(x + y)**2 - f(x + y),
        f(x)**2 - 2*f(x)*f(y) - 2*f(x)*f(x - y) + 2*f(x) + f(y)**2 - 2*f(y)*f(x - y) + 2*f(y) + f(x - y)**2 - f(x - y) - f(x + y),
        2*f(x)**2 + 2*f(x)*f(y) + f(x)*f(x - y) + f(x)*f(x + y) + 2*f(x) + 2*f(y)*f(x - y) + 2*f(y)*f(x + y) + 2*f(y) - f(x - y)**2 - 2*f(x - y)*f(x + y) - f(x - y) - f(x + y)**2 - f(x + y),
        2*f(x)**2 - f(x)*f(x - y) - f(x)*f(x + y) - 2*f(x) - 2*f(y)**2 + f(y)*f(x - y) + f(y)*f(x + y) - 2*f(y) + f(x - y) + f(x + y),
        2*f(x)**2 + 2*f(x)*f(y) - f(x)*f(x - y) + f(x)*f(x + y) - 2*f(x) + 2*f(y)*f(x + y) - 2*f(y) - f(x - y)*f(x + y) + f(x - y) - f(x + y)**2 + f(x + y),
        2*f(x)**2 + 2*f(x)*f(y) + f(x)*f(x - y) - f(x)*f(x + y) + 2*f(x) + 2*f(y)*f(x - y) + 2*f(y) - f(x - y)**2 - f(x - y)*f(x + y) - f(x - y) - f(x + y),
        2*f(x)*f(y) + 2*f(x)*f(x + y) + 2*f(x) + 2*f(y)**2 - f(y)*f(x - y) + f(y)*f(x + y) + 2*f(y) - f(x - y)*f(x + y) - f(x - y) - f(x + y)**2 - f(x + y),
        2*f(x)**2 - f(x)*f(x - y) - f(x)*f(x + y) + 2*f(x) - 2*f(y)**2 + f(y)*f(x - y) + f(y)*f(x + y) + 2*f(y) - f(x - y) - f(x + y),
        2*f(x)*f(y) - 2*f(x) + 2*f(y)**2 - f(y)*f(x - y) - f(y)*f(x + y) - 2*f(y) + f(x - y) + f(x + y),
        f(x)**2 - 2*f(x)*f(y) + 2*f(x)*f(x + y) + 2*f(x) + f(y)**2 + 2*f(y)*f(x + y) + 2*f(y) - 2*f(x - y)*f(x + y) - f(x - y) - f(x + y)**2 - f(x + y),
    ]
    # fmt: on
    properties = [Property(p) for p in properties]
    eq_classes = create_equivalence_classes(properties)
    print("eq_classes:")
    for cls in eq_classes:
        print(cls)
    assert len(eq_classes) == len(properties)


if __name__ == "__main__":  # noqa E123
    from sympy import Symbol

    test_create_equivalence_classes()
    test_squared_function()
