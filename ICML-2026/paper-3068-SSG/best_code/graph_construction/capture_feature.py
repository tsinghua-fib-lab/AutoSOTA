import ast
import inspect
import textwrap
import types
import torch
import time

class ExitToMainException(Exception):
    """Custom exception used to exit directly to the main function."""
    def __init__(self, message="", exit_code=0, data=None):
        self.message = message
        self.exit_code = exit_code
        self.data = data
        super().__init__(message)

def _get_assigned_names(node):
    names = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]
    else:
        targets = []

    def collect(target):
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                collect(elt)
        # Ignore attribute assignments such as self.x.
    for t in targets:
        collect(t)
    return names

class InjectCaptureTransformer(ast.NodeTransformer):
    def __init__(self, target_names, injected_name, mode="first"):
        super().__init__()
        self.target_names = set(target_names)
        self.injected_name = injected_name
        self.mode = mode  # "first" or "last"

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # === Step 1: scan assignments and collect statements by variable. ===
        assignments = {name: [] for name in self.target_names}

        for stmt in node.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                assigned_names = _get_assigned_names(stmt)
                for name in assigned_names:
                    if name in self.target_names:
                        assignments[name].append(stmt)

        # === Step 2: determine after which statement capture should be inserted for each variable. ===
        # Record where to insert capture calls and which names should be captured there.
        insert_info = {}  # stmt_node -> set of names to capture after this stmt

        for name in self.target_names:
            stmts = assignments[name]
            if not stmts:
                continue
            target_stmt = stmts[0] if self.mode == "first" else stmts[-1]
            if target_stmt not in insert_info:
                insert_info[target_stmt] = set()
            insert_info[target_stmt].add(name)

        # === Step 3: build a new function body and insert capture calls after selected statements. ===
        new_body = []
        for stmt in node.body:
            new_body.append(stmt)
            if stmt in insert_info:
                # Get all variable names that should be captured after this statement.
                names_to_capture = sorted(insert_info[stmt])  # Sort for deterministic output.
                for name in names_to_capture:
                    call = ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id=self.injected_name, ctx=ast.Load()),
                            args=[
                                ast.Constant(value=name),
                                ast.Name(id=name, ctx=ast.Load())
                            ],
                            keywords=[]
                        )
                    )
                    new_body.append(call)

        node.body = new_body
        return node




def instrument_forward_and_capture(module, var_names, mode="last", capture_until_num=None):
    if isinstance(var_names, str):
        var_names = [var_names]
    var_names = list(var_names)
    original_forward = module.forward
    features = {}
    features['_target_names'] = var_names  # Passed to the capture function to initialize counters.

    def _make_capture(features_dict, mode_flag, capture_until_num):
        target_names = features_dict.get('_target_names', [])
        capture_count = {name: 0 for name in target_names}

        def _capture(name, value):
            try:
                is_tensor = isinstance(value, torch.Tensor)
            except Exception:
                is_tensor = False
            saved = value.detach().clone() if is_tensor else value

            # Update the latest value.
            try:
                features_dict[name] = torch.cat([features_dict[name],saved], dim=0)
            except KeyError:
                features_dict[name] = saved
            except TypeError:
                features_dict[name] = saved


            if 'start_time' not in features_dict:
                features_dict['start_time'] = time.time()

            # Increment the count.
            if name not in capture_count:
                capture_count[name] = 0
            capture_count[name] += 1
            # print(f"-------------------------------------capture_count[{name}]:{capture_count[name]}-------------------------------------")
            # Check whether the cycle threshold has been reached.
            if capture_until_num and (capture_count[name] % capture_until_num == 0):
                raise ExitToMainException(
                    message=f"[Capture#{capture_count[name]}] '{name}' captured {capture_until_num}x cycle -> Break!",
                    exit_code=0,
                    data=saved
                )

            return value

        # === Expose the counter as an attribute for external access. ===
        _capture.__dict__['capture_count'] = capture_count
        _capture.__dict__['reset_counter'] = lambda names=None: _reset_count(names)
        
        def _reset_count(names=None):
            """Reset the counter.
            Args:
                names: str or list of str, variable names to reset; None means all variables
            """
            if names is None:
                vars_to_reset = target_names
            elif isinstance(names, str):
                vars_to_reset = [names]
            else:
                vars_to_reset = names
            
            for name in vars_to_reset:
                if name in capture_count:
                    capture_count[name] = 0

        return _capture

    capture_func = _make_capture(features, mode, capture_until_num)

    try:
        src = inspect.getsource(original_forward)
    except Exception as e:
        raise RuntimeError(f"Cannot get source: {e}")

    src = textwrap.dedent(src)
    try:
        mod_ast = ast.parse(src)
    except SyntaxError as e:
        raise RuntimeError(f"Failed to parse source: {e}")

    g = original_forward.__globals__
    base_injected = "_capture_injected"
    injected_name = base_injected
    if injected_name in g:
        i = 1
        while f"{base_injected}_{i}" in g:
            i += 1
        injected_name = f"{base_injected}_{i}"

    # Pass mode to the transformer.
    transformer = InjectCaptureTransformer(var_names, injected_name, mode=mode)
    mod_ast = transformer.visit(mod_ast)
    ast.fix_missing_locations(mod_ast)

    g[injected_name] = capture_func
    prev_global_forward = g.get(original_forward.__name__, None)

    filename = inspect.getsourcefile(original_forward) or "<string>"
    compiled = compile(mod_ast, filename, "exec")
    exec(compiled, g)

    new_forward = g.get(original_forward.__name__)
    if new_forward is None or not isinstance(new_forward, types.FunctionType):
        if injected_name in g and g[injected_name] is capture_func:
            del g[injected_name]
        raise RuntimeError("Failed to create new forward function.")

    bound_new_forward = new_forward.__get__(module, module.__class__)
    module.forward = bound_new_forward

    if prev_global_forward is None:
        if g.get(original_forward.__name__) is new_forward:
            del g[original_forward.__name__]
    else:
        g[original_forward.__name__] = prev_global_forward

    def restore():
        module.forward = original_forward
        if injected_name in g and g[injected_name] is capture_func:
            del g[injected_name]

    # Expose reset_counter.
    def reset_counter(names=None):
        """Reset the capture counter."""
        if hasattr(capture_func, 'reset_counter'):
            capture_func.reset_counter(names)

    return features, restore, reset_counter

########################################
class C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(10, 10)

    def forward(self, x):
        h = self.proj(x)
        tmp = torch.relu(h)   # Capture tmp even though it is not returned.
        print("first tmp", tmp)
        tmp = tmp * 2         # Modify tmp again.
        # print("second tmp", tmp)
        tmp = tmp -10
        print("last tmp", tmp)
        out = tmp + 1
        return out

class B(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.C = C()
    def forward(self, x):
        return self.C(x)

class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.B = B()
    def forward(self, x):
        return self.B(x)

if __name__=="__main__":
    # ====== Use the instrumentation helper. ======
    model = A()
    x = torch.randn(2, 10)

    # --- Capture the first assignment. ---
    features_first, restore_first = instrument_forward_and_capture(
        model.B.C, ["tmp"], mode="first", capture_until_num=3
    )
    out = model(x)
    print("111 Captured mode=First:", features_first["tmp"])

    x = torch.randn(2, 10)
    out = model(x)
    print("222 Captured mode=First:", features_first["tmp"])

    x = torch.randn(2, 10)
    out = model(x)
    print("333 Captured mode=First:", features_first["tmp"])

    x = torch.randn(2, 10)
    out = model(x)
    print("444 Captured mode=First:", features_first["tmp"])

    x = torch.randn(2, 10)
    out = model(x)
    print("555 Captured mode=First:", features_first["tmp"])

    restore_first()


    # --- Capture the last assignment. ---
    features_last, restore_last = instrument_forward_and_capture(
        model.B.C, ["tmp"], mode="last"
    )
    out = model(x)
    print("Captured mode=Last:", features_last["tmp"])
    restore_last()
