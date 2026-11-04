# This source is concatenated into the runtime metadata probe's globals.
if runtime_adapter == "vllm_pd":
    software["nixl"] = distribution_version("nixl")
    ucx_name = ctypes.util.find_library("ucp") or "libucp.so.0"
    ucx = ctypes.CDLL(ucx_name)
    get_ucx_version = ucx.ucp_get_version_string
    get_ucx_version.restype = ctypes.c_char_p
    ucx_text = get_ucx_version().decode("utf-8")
    ucx_match = re.search(r"\d+(?:\.\d+){1,3}", ucx_text)
    if not ucx_match:
        raise RuntimeError(f"Could not parse UCX version from {ucx_text!r}")
    software["ucx"] = ucx_match.group(0)
