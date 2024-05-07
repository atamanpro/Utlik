from fastapi import FastAPI
from tool_functions.router import router


app = FastAPI(
    title="Tools API",
    description="""
    You can be able to do the following:
    **first_endpoint** - get some data from ..
    **second_endpoint** - post some data to ..
    """,
    version="0.0.1",
    docs_url="/tools/docs",
    redoc_url="/tools/redoc",
    openapi_url="/tools/openapi.json")

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=82)