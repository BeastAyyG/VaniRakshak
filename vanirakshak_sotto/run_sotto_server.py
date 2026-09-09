import uvicorn

if __name__ == "__main__":
    print("Launching VaniRakshak Kavach Sotto Edition (OpenVINO) on http://127.0.0.1:8082 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8082, reload=False)
