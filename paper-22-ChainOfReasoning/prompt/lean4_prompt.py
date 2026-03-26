# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

prompt = fr"""Now please translate the formal solution in Lean 4 following the instruction below. Please write the corresponding solution in Lean 4 (indicated by ``Formal proof in Lean 4: '') given the ``# Problem: '' and ``# Informal proof: '', filling in the ``# Formal proof in Lean 4: '' section.

You must respond in the following format: 

# Problem:  ...

# Informal proof: ...

# Formal proof in Lean 4: 
```lean4
...

```

Here are examples you may refer to:

---

# Problem: In a room, there are 4 chairs along each side of a square table. The length of the side of the table is 1 meter. What is the total length of all sides of the table?

# Informal proof: A square has four equal sides. Given that each side of the table is 1 meter, the total length of all sides of the table is 1 meter * 4 sides = 4 meters.

# Formal proof in Lean 4: 
```lean4\ndef side_length := 1\ndef number_of_sides := 4\ndef total_length := side_length * number_of_sides\n#reduce total_length   -- returns 4\n```

---

# Problem: Dave breaks 2 guitar strings per night when playing live.  If he performs 6 shows a week for 12 weeks, how many guitar strings will he need to replace?

# Informal proof: To determine the total number of guitar strings Dave will need to replace, we need to calculate the number of strings he breaks per week and then multiply that by the number of weeks he performs. Here’s the step-by-step process:\n\n1. Calculate the number of guitar strings Dave breaks per week.\n2. Multiply this weekly amount by the total number of weeks he is performing.\n\nLet's proceed with this calculation using Python:\n```python\n# Constants\nstrings_broken_per_night = 2\nshows_per_week = 6\nweeks = 12\n\n# Calculate total guitar strings broken per week\nstrings_broken_per_week = strings_broken_per_night * shows_per_week\n\n# Calculate total guitar strings broken over the period of 12 weeks\ntotal_strings_broken = strings_broken_per_week * weeks\nprint(total_strings_broken)\n```\n```output\n144\n```\nDave will need to replace \\(\\boxed{144}\\) guitar strings over the course of 12 weeks.

# Formal proof in Lean 4: 
```lean4\ndef strings_broken_per_night := 2\ndef shows_per_week := 6\ndef weeks := 12\n\ndef strings_broken_per_week := strings_broken_per_night * shows_per_week\ndef total_strings_broken := strings_broken_per_week * weeks\n\n#reduce total_strings_broken  -- returns 144\n```

---

# Problem: Convert the binary number $101_{(2)}$ to a decimal number.

# Informal proof: To solve the problem of converting the binary number \\(101_{(2)}\\) to a decimal number, we can follow these steps:\n\n1. Understand binary to decimal conversion: Each digit in a binary number represents an increasing power of 2, starting from the rightmost digit, which represents \\(2^0\\).\n\n2. Write a Python code using sympy to perform the conversion.\n\nThe binary number \\(101_{(2)}\\) can be converted to decimal as follows:\n- \\(1 \\times 2^2 + 0 \\times 2^1 + 1 \\times 2^0\\)\n\nLet's implement this in Python.\n\n```python\n# Define the binary number as a string\nbinary_num = '101'\n\n# Initialize the decimal number\ndecimal_num = 0\n\n# Iterate over the binary number\nfor i, digit in enumerate(reversed(binary_num)):\n    decimal_num += int(digit) * (2 ** i)\n\n# Print the result\nprint(decimal_num)\n```\n```output\n5\n```\nThe decimal equivalent of the binary number \\(101_{(2)}\\) is \\(\\boxed{5}\\).

# Formal proof in Lean 4: 
```lean4\ndef binary_to_decimal : List Nat → Nat\n| []       => 0\n| (b :: bs) => b + 2 * binary_to_decimal bs\n\ndef binary_101 := [1, 0, 1]\n\ndef decimal_value := binary_to_decimal binary_101.reverse\n\n#reduce decimal_value  -- returns 5\n```

---

# Problem: A bag contains 5 red, 6 green, 7 yellow, and 8 blue jelly beans.   A jelly bean is selected at random.  What is the probability that it is blue?

# Informal proof: To determine the probability of selecting a blue jelly bean from the bag, we'll use the following method:\n\n1. Calculate the total number of jelly beans.\n2. Determine the number of blue jelly beans.\n3. Use the formula for probability:\n\n\\[ P(\\text{{Blue}}) = \\frac{{\\text{{Number of Blue Jelly Beans}}}}{{\\text{{Total Number of Jelly Beans}}}} \\]\n\nLet's calculate this step-by-step using Python.\n```python\n# Numbers of different colored jelly beans in the bag\nnum_red = 5\nnum_green = 6\nnum_yellow = 7\nnum_blue = 8\n\n# Total number of jelly beans\ntotal_jelly_beans = num_red + num_green + num_yellow + num_blue\n\n# Number of blue jelly beans\nnum_blue_jelly_beans = num_blue\n\n# Probability of picking a blue jelly bean\nprobability_blue = num_blue_jelly_beans / total_jelly_beans\nprint(probability_blue)\n```\n```output\n0.3076923076923077\n```\nThe probability of randomly selecting a blue jelly bean from the bag is \n\n\\[\n\\boxed{0.3077}\n\\]\n\nor 30.77%.

# Formal proof in Lean 4: 
```lean4\ndef num_red := 5\ndef num_green := 6\ndef num_yellow := 7\ndef num_blue := 8\n\ndef total_jelly_beans := num_red + num_green + num_yellow + num_blue\ndef probability_blue := num_blue.toFloat / total_jelly_beans.toFloat\n\n#eval probability_blue  -- returns 0.3076923076923077\n```

---

# Problem: The sum of the first $n$ terms of an arithmetic sequence is given by $n^{2} + 5n$. Find the tenth term of the sequence.

# Informal proof: To find the tenth term of the arithmetic sequence whose sum of the first \\(n\\) terms is \\(n^2 + 5n\\), we need to perform the following steps:\n\n1. **Express the sum of the first \\(n\\) terms \\(S_n\\)**:\n   \\[\n   S_n = n^2 + 5n\n   \\]\n\n2. **Use the sum of the first \\(n\\) terms to find the \\(n\\)-th term \\(a_n\\)** of the sequence**:\n   The \\(n\\)-th term \\(a_n\\) can be found using the formula:\n   \\[\n   a_n = S_n - S_{{n-1}}\n   \\]\n   where \\(S_{{n-1}}\\) is the sum of the first \\((n-1)\\) terms.\n\n3. **Substitute the given sum formula**:\n   \\[\n   S_{{n-1}} = (n-1)^2 + 5(n-1)\n   \\]\n   \\[\n   a_n = (n^2 + 5n) - ((n-1)^2 + 5(n-1))\n   \\]\n\n4. **Simplify to find the expression for \\(a_n\\)**:\n   This involves substituting and simplifying the expressions.\n\nLet's use Python to perform these steps and find the tenth term \\(a_{10}\\).\n```python\n# Define the expressions for S_n and S_(n-1)\ndef S_n(n):\n    return n**2 + 5*n\n\ndef S_n_minus_1(n):\n    return (n-1)**2 + 5*(n-1)\n\n# Calculate the nth term a_n\ndef a_n(n):\n    return S_n(n) - S_n_minus_1(n)\n\n# Find the 10th term a_10\na_10 = a_n(10)\nprint(a_10)\n```\n```output\n24\n```\nThe tenth term \\(a_{10}\\) of the arithmetic sequence is \\(\\boxed{24}\\).

# Formal proof in Lean 4: 
```lean4\ndef S (n : Nat) : Nat := n^2 + 5 * n\n\ndef a (n : Nat) : Nat := S n - S (n - 1)\n\ndef a_10 := a 10\n\n#eval a_10  -- returns 24\n```

---

# Problem: The sum of the first $n$ terms of an arithmetic sequence is given by $n^{2} + 5n$. Find the tenth term of the sequence.

# Informal proof: To find the tenth term of the arithmetic sequence whose sum of the first \\(n\\) terms is \\(n^2 + 5n\\), we need to perform the following steps:\n\n1. **Express the sum of the first \\(n\\) terms \\(S_n\\)**:\n   \\[\n   S_n = n^2 + 5n\n   \\]\n\n2. **Use the sum of the first \\(n\\) terms to find the \\(n\\)-th term \\(a_n\\)** of the sequence**:\n   The \\(n\\)-th term \\(a_n\\) can be found using the formula:\n   \\[\n   a_n = S_n - S_{{n-1}}\n   \\]\n   where \\(S_{{n-1}}\\) is the sum of the first \\((n-1)\\) terms.\n\n3. **Substitute the given sum formula**:\n   \\[\n   S_{{n-1}} = (n-1)^2 + 5(n-1)\n   \\]\n   \\[\n   a_n = (n^2 + 5n) - ((n-1)^2 + 5(n-1))\n   \\]\n\n4. **Simplify to find the expression for \\(a_n\\)**:\n   This involves substituting and simplifying the expressions.\n\nLet's use Python to perform these steps and find the tenth term \\(a_{10}\\).\n```python\n# Define the expressions for S_n and S_(n-1)\ndef S_n(n):\n    return n**2 + 5*n\n\ndef S_n_minus_1(n):\n    return (n-1)**2 + 5*(n-1)\n\n# Calculate the nth term a_n\ndef a_n(n):\n    return S_n(n) - S_n_minus_1(n)\n\n# Find the 10th term a_10\na_10 = a_n(10)\nprint(a_10)\n```\n```output\n24\n```\nThe tenth term \\(a_{10}\\) of the arithmetic sequence is \\(\\boxed{24}\\).

# Formal proof in Lean 4: 
```lean4\ndef S (n : Nat) : Nat := n^2 + 5 * n\n\ndef a (n : Nat) : Nat := S n - S (n - 1)\n\ndef a_10 := a 10\n\n#eval a_10  -- returns 24\n```

---

# Problem: If \\( x \\) is a number less than \\(-2\\), which of the following expressions has the least value?\n(A) \\( x \\)\n(B) \\( x + 2 \\)\n(C) \\( \\frac{1}{2}x \\)\n(D) \\( x - 2 \\)\n(E) \\( 2x \\)

# Informal proof: To solve this problem, we need to compare the values of the given expressions for \\( x < -2 \\):\n\n(A) \\( x \\)\n(B) \\( x + 2 \\)\n(C) \\( \\frac{1}{2}x \\)\n(D) \\( x - 2 \\)\n(E) \\( 2x \\)\n\nWe'll analyze each expression to determine which one has the least value when \\( x \\) is less than \\(-2\\).\n\n1. **Expression Analysis:**\n   - For any \\( x < -2 \\):\n     - \\( x + 2 \\) will be greater than \\( x \\) because adding 2 makes it less negative.\n     - \\( \\frac{1}{2}x \\) will be less negative than \\( x \\) because halving a negative number makes it closer to zero.\n     - \\( x - 2 \\) will be less than \\( x \\) because subtracting 2 makes it more negative.\n     - \\( 2x \\) will be more negative than \\( x \\) because multiplying a negative number by 2 makes it further from zero.\n\n2. **Determine Least Value:**\n   We can evaluate each expression for a specific \\( x \\) less than \\(-2\\) to find which has the least value. Let's pick \\( x = -3 \\) for simplicity.\n\nLet's implement this reasoning in Python to confirm:\n\n```python\nimport sympy as sp\n\n# Define the variable x\nx = sp.symbols('x')\n\n# Define the expressions\nexpressions = {{\n    'A': x,\n    'B': x + 2,\n    'C': sp.Rational(1, 2) * x,\n    'D': x - 2,\n    'E': 2 * x\n}}\n\n# Define a value for x less than -2\nx_value = -3\n\n# Substitute the value into each expression and evaluate\nevaluations = {{key: expr.subs(x, x_value) for key, expr in expressions.items()}}\n\n# Find the expression with the least value\nleast_value_key = min(evaluations, key=evaluations.get)\nleast_value = evaluations[least_value_key]\n\nprint(f\"The expression with the least value is {{least_value_key}} with a value of {{least_value}}\")\n```\n```output\nThe expression with the least value is E with a value of -6\n```\nBased on the evaluations, the expression with the least value when \\( x < -2 \\) is \\( 2x \\), which corresponds to option (E). This is confirmed by the output when \\( x = -3 \\):\n\n- \\( x \\) evaluates to \\(-3\\)\n- \\( x + 2 \\) evaluates to \\(-1\\)\n- \\( \\frac{1}{2}x \\) evaluates to \\(-1.5\\)\n- \\( x - 2 \\) evaluates to \\(-5\\)\n- \\( 2x \\) evaluates to \\(-6\\)\n\nTherefore, the expression with the least value is \\(\\boxed{{2x}}\\) (option E).

# Formal proof in Lean 4: 
```lean4\ndef expression_A (x : Int) : Int := x\ndef expression_B (x : Int) : Int := x + 2\ndef expression_C (x : Int) : Int := x / 2\ndef expression_D (x : Int) : Int := x - 2\ndef expression_E (x : Int) : Int := 2 * x\n\ndef least_value_expression (x : Int) : String :=\n  let values := [\n    (\"A\", expression_A x),\n    (\"B\", expression_B x),\n    (\"C\", expression_C x),\n    (\"D\", expression_D x),\n    (\"E\", expression_E x)\n  ]\n  values.foldl (λ acc val => if val.snd < acc.snd then val else acc) (\"\", x)\n  |>.fst\n\ndef x := -3\ndef result := least_value_expression x\n\n#eval result  -- returns \"E\"\n```

---

# Problem: The integer 42 is:\n(D) divisible by 7.

# Informal proof: To check if the integer 42 is divisible by 7, we can use Python's modulus operator `%`. If `42 % 7` equals 0, then 42 is divisible by 7.\n\nLet's break this down into steps and write the Python code to solve it:\n\n1. Use the modulus operator `%` to check if 42 is divisible by 7.\n2. Print the result.\n\nHere is the Python code to achieve this:\n\n```python\n# Check if 42 is divisible by 7\ndivisible_by_7 = 42 % 7 == 0\nprint(divisible_by_7)\n```\n```output\nTrue\n```\nThe output confirms that 42 is divisible by 7. \n\nThus, the statement \"The integer 42 is divisible by 7\" is \\(\\boxed{True}\\).

# Formal proof in Lean 4: 
```lean4\ndef is_divisible_by_7 (n : Int) : Bool :=\n  n % 7 == 0\n\ndef result := is_divisible_by_7 42\n\n#eval result  -- returns true\n```

---

# Problem: A 3-element set \\( S \\) has the sum of the elements of all its subsets equal to 2012 (consider the sum of elements of the empty set to be zero). What is the sum of the elements of \\( S \\) equal to?

# Informal proof: To solve this problem, we need to analyze the properties of a set \\( S = \\{{a, b, c\\}} \\). The sum of the elements of all subsets of \\( S \\) includes the sum of the empty set, individual elements, and all possible combinations of these elements.\n\nLet's list all the subsets of \\( S \\):\n1. The empty set: \\( \\{{\\}} \\)\n2. Single-element subsets: \\( \\{{a\\}}, \\{{b\\}}, \\{{c\\}} \\)\n3. Two-element subsets: \\( \\{{a, b\\}}, \\{{a, c\\}}, \\{{b, c\\}} \\)\n4. The full set: \\( \\{{a, b, c\\}} \\)\n\nNow sum up the elements of all subsets:\n\\[\n\\begin{{align*}}\n\\text{{Sum}} &= 0 \\quad (\\text{{empty set}})\\\\\n&+ (a + b + c) \\quad (\\text{{single-element subsets}})\\\\\n&+ (a + b + a + c + b + c) \\quad (\\text{{two-element subsets}})\\\\\n&+ (a + b + c) \\quad (\\text{{full set}})\n\\end{{align*}}\n\\]\n\nRewriting the sum gives:\n\\[\n\\text{{Sum}} = 3(a + b + c) + 3(a + b + c) + (a + b + c) = 7(a + b + c)\n\\]\n\nAccording to the problem, the sum of all these subsets equals 2012:\n\\[\n7(a + b + c) = 2012\n\\]\n\nTo find \\( a + b + c \\), we simply divide both sides by 7:\n\\[\na + b + c = \\frac{2012}{7}\n\\]\n\nLet's perform the division to get the final answer.\n```python\n# Calculate the sum of elements of S by dividing 2012 by 7\nsum_of_elements = 2012 / 7\nprint(sum_of_elements)\n```\n```output\n287.42857142857144\n```\nSince we are dealing with sets of integers and the result (287.42857142857144) is not an integer, we need to double-check our calculations and verify if there was any mistake in our reasoning steps or assumptions.\n\nLet's reconsider the problem statement and our breakdown once again:\n\nEach element in \\( S = \\{{a, b, c\\}} \\) appears in subsets formed with it alone, with one more element, and then in the full set. Each element appears in total in exactly half of the subsets (since for every subset without the element, there is a corresponding subset including it. Specifically, for a set \\( S = \\{{a, b, c\\}} \\), there are \\( 2^n - 1 = 7 \\) such subsets when considering non-empty).\n\nTherefore, each element \\( a, b, \\) and \\( c \\) contributes to the total sum exactly \\( 2^2 = 4 \\) times (once for itself, twice when paired with another element, and once in the complete set).\nSo the sum of all contributions equals \\( 4(a + b + c) = 2012 \\).\n\nThus:\n\n\\[ 4(a + b + c) = 2012 \\]\n\nSolving for \\( a + b + c \\):\n\n\\[ a + b + c = \\frac{2012}{4} = 503 \\]\n\nLet's compute \\( 503 \\) to verify.\n```python\n# Correcting the extra term, we actually sum each element 4 times:\n# Re-compute based on the corrected formula\ncorrected_sum_of_elements = 2012 / 4\nprint(corrected_sum_of_elements)\n```\n```output\n503.0\n```\nThe correct sum of the elements of \\( S \\) is indeed:\n\n\\[\n\\boxed{503}\n\\]

# Formal proof in Lean 4: 
```lean4\n-- Define the sum of all subsets of a set with three elements\ndef sum_of_subsets (a b c : Int) : Int :=\n  let subsets := [\n    0,        -- empty set\n    a, b, c,  -- single-element subsets\n    a + b, a + c, b + c,  -- two-element subsets\n    a + b + c  -- full set\n  ]\n  subsets.foldl (· + ·) 0  -- calculate the total sum of all subsets\n\n-- Calculate the sum of the elements based on the total sum of subsets\ndef sum_of_elements (total_sum : Int) : Int :=\n  total_sum / 4  -- each element appears in exactly half of the subsets\n\n-- Define the total sum of all subsets as given in the problem\ndef total_sum := 2012\n\n-- Calculate the result for the sum of the elements of the set\ndef result := sum_of_elements total_sum\n\n-- Output the result\n#eval result  -- returns 503\n```

---

# Problem: {problem}

# Informal proof: {informal_proof}

# Formal proof in Lean 4: 
```lean4
...

```
    """
