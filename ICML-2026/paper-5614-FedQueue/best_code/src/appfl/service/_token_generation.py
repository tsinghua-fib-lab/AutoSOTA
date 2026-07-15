import argparse
import platform
import globus_sdk
from globus_sdk import NativeAppAuthClient
from globus_sdk.scopes import AuthScopes, ComputeScopes
from globus_compute_sdk import Executor, Client
from globus_compute_sdk.sdk.login_manager import AuthorizerLoginManager

argparser = argparse.ArgumentParser()
argparser.add_argument("--login", action="store_true")
argparser.add_argument("--compute_token", required=False)
argparser.add_argument("--openid_token", required=False)
argparser.add_argument("--endpoint_id", required=False)
args = argparser.parse_args()


if args.login:
    auth_client = NativeAppAuthClient(
        client_id="3de1d1a4-cd37-4d5f-a775-01d9c430330b",
        app_name="APPFL",
    )

    auth_client.oauth2_start_flow(
        refresh_tokens=False,
        requested_scopes=[ComputeScopes.all, AuthScopes.openid],
        prefill_named_grant=platform.node(),
    )
    print(
        "Please authenticate with Globus here\n"
        "------------------------------------\n"
        f"{auth_client.oauth2_get_authorize_url(query_params={'prompt': 'login'})}\n"
        "------------------------------------\n"
    )

    auth_code = input(
        "Please enter the authorization code you get after login here: "
    ).strip()
    token_response = auth_client.oauth2_exchange_code_for_tokens(auth_code)

    print(token_response)

if (
    (args.login) or (args.compute_token is not None and args.openid_token is not None)
) and args.endpoint_id is not None:
    compute_token = (
        args.compute_token
        if args.compute_token is not None
        else token_response.by_resource_server[ComputeScopes.resource_server][
            "access_token"
        ]
    )
    openid_token = (
        args.openid_token
        if args.openid_token is not None
        else token_response.by_resource_server[AuthScopes.resource_server][
            "access_token"
        ]
    )
    compute_auth = globus_sdk.AccessTokenAuthorizer(compute_token)
    openid_auth = globus_sdk.AccessTokenAuthorizer(openid_token)
    compute_login_manager = AuthorizerLoginManager(
        authorizers={
            ComputeScopes.resource_server: compute_auth,
            AuthScopes.resource_server: openid_auth,
        }
    )
    compute_login_manager.ensure_logged_in()
    gc = Client(login_manager=compute_login_manager)

    def double(x):
        return x * 2

    with Executor(endpoint_id=args.endpoint_id, client=gc) as gce:
        fut = gce.submit(double, 7)
        print(fut.result())

    print("Endpoint Token Test Successful!")
