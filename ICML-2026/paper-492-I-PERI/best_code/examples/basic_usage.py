def _example_usage_check():
    """Petit auto-test local — ne s'exécute que si le module est invoqué directement."""
    if True:
        # create small linear SCM: U -> X -> Y
        v = ['X', 'Y']
        u = ['U']
        coeffs = {('U', 'X'): 2.0, ('X', 'Y'): 3.0}
        scm = LinearSCM(v=v, u=u, coefficients=coeffs)
        df = scm.generate_data(100, include_latent=False, seed=0)
        print(df)