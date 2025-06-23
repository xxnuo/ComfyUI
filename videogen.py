import uvicorn

if __name__ == "__main__":
    uvicorn.run("webui.server:app", host="0.0.0.0", port=3000)
