from appfl.login_manager.globus import GlobusLoginManager


def auth():
    """Command line interface for authenticating with Globus Auth."""
    prompt = "Please select whether you are authenticating for the federated learning server or client (Enter 1 or 2)"
    while True:
        mode = input(
            f"{prompt}\n"
            f"{'-' * len(prompt)}\n"
            "1. Federated learning server\n"
            "2. Federated learning client\n"
            f"{'-' * len(prompt)}\n"
        )
        if mode in ("1", "2"):
            f"{'-' * len(prompt)}\n"
            break
        else:
            print(f"Invalid input, please try again\n{'-' * len(prompt)}\n")
    is_fl_server = mode == "1"
    login_manager = GlobusLoginManager(is_fl_server=is_fl_server)
    if login_manager.ensure_logged_in():
        action = input(
            f"You have already logged in as a federated learning {'server' if is_fl_server else 'client'}\n"
            "You can either logout (1), change to another account (2), or just exit (3)\n"
            f"{'-' * len(prompt)}\n"
            "1. Logout\n"
            "2. Change to another account\n"
            "3. Exit\n"
            f"{'-' * len(prompt)}\n"
        )
        if action not in ["1", "2", "3"]:
            print("Invalid input, exiting...")
            return
        if action == "1":
            login_manager.logout()
            print("Successfully logged out\n")
            return
        elif action == "2":
            login_manager.logout()
            login_manager.ensure_logged_in()
            print(
                f"{'-' * len(prompt)}\n"
                f"Successfully logged in as a federated learning {'server' if is_fl_server else 'client'}\n"
            )
    else:
        print(
            f"{'-' * len(prompt)}\n"
            f"Successfully logged in as a federated learning {'server' if is_fl_server else 'client'}\n"
        )


def appfl_globus_auth():
    auth()
