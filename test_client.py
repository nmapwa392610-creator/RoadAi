ok, err = check_file(uploaded_file)
if not ok:
    import magic
    header = uploaded_file.file.read(4096)
    print("Magic MIME:", magic.from_buffer(header, mime=True))