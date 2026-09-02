if __name__ == "__main__":
    from globus_compute_sdk import Client
    from triton_hep_client import triton_rpc
    func_uuid = Client().register_function(triton_rpc)
    print(f"Registered {triton_rpc.__name__}: {func_uuid=}")
