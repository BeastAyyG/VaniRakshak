import uvicorn

if __name__ == "__main__":
    print("Launching VaniRakshak Kavach IndicConformer Edition (AI4Bharat ASR) on http://127.0.0.1:8084 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8084, reload=False)
