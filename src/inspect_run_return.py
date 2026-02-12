from mcp_server import mcp
import asyncio
import inspect

async def main():
    try:
        app = mcp.streamable_http_app()
        sm = mcp.session_manager
        
        ret = sm.run()
        print(f"Return type: {type(ret)}")
        
        if hasattr(ret, '__aenter__'):
            print("It IS an async context manager!")
        else:
            print("It is NOT an async context manager directly.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
