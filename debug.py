import os

from daytona import AsyncDaytona, CreateSandboxFromImageParams, DaytonaConfig, ExecuteResponse, Image, Resources

from src.utils import apply_patch, create_evaluation_script, create_run_command

environment_keys: dict[str, str] = {
    "DAYTONA_API_KEY": os.getenv("DAYTONA_API_KEY") or "",
    "DAYTONA_API_URL": os.getenv("DAYTONA_API_URL") or "",
    "DAYTONA_TARGET": os.getenv("DAYTONA_TARGET") or "",
}

daytona = AsyncDaytona(
    config=DaytonaConfig(
        api_key=environment_keys["DAYTONA_API_KEY"],
        api_url=environment_keys["DAYTONA_API_URL"],
        target=environment_keys["DAYTONA_TARGET"],
    )
)


async def main():
    sandbox = await daytona.create()
    with open("patch.diff", "rb") as patch_file:
        patch_bytes = patch_file.read()

    sandbox = await daytona.create(
        CreateSandboxFromImageParams(
            name="debugging-claude",
            image=Image.base("ghcr.io/epoch-research/swe-bench.eval.x86_64.astropy__astropy-12907:latest"),
            network_block_all=False,
            resources=Resources(
                cpu=4,
                memory=8,
                disk=10,
            ),
        ),
        timeout=360,
    )

    await sandbox.fs.upload_file(
        patch_bytes,
        "/tmp/patch.diff",
    )

    await sandbox.process.exec(
        command="git reset --hard HEAD && git clean -fd",
        cwd="/testbed",
    )

    await apply_patch(sandbox, "/tmp/patch.diff")

    evaluation_script: str = create_evaluation_script("astropy__astropy-12907", sandbox.id)

    await sandbox.fs.upload_file(
        evaluation_script.encode("utf-8"),
        "/root/eval.sh",
    )

    run_command: str = create_run_command(sandbox.id)

    result: ExecuteResponse = await sandbox.process.exec(
        command=run_command,
    )

    print(result.result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
