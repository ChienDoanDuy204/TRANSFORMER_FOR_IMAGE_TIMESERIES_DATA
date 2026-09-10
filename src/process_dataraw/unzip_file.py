# function to unzip data raw
def UnzipFile(path: str = None, output: str = None) -> dict:
    import zipfile
    """
    Input:
    path: đường dẫn file zip cần giải nén
    output: thư mục output sẽ chứa file sau khi giải nén
    """

    return_result = {
        "code": 200,
        "message": "Success!"
    }
    try:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(output)
            file_names = zip_ref.namelist()
            return_result["file_names"] = file_names
    except Exception as e:
        return_result["code"] = 500
        return_result["message"] = f"Error: {e}"
    
    return return_result