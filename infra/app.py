from typing import Any

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    Stack,
)
from aws_cdk import (
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_ecr as ecr,
)
from aws_cdk import (
    aws_ecs as ecs,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk.aws_ecr_assets import Platform
from constructs import Construct


class FastApiStack(Stack):
    _SERVICE_NAME: str = "Swebench"

    def __init__(self, scope: Construct, id: str, **kwargs: Any):
        super().__init__(scope, id, **kwargs)

        vpc = ec2.Vpc(self, f"{self._SERVICE_NAME}Vpc", max_azs=2)

        cluster = ecs.Cluster(self, f"{self._SERVICE_NAME}Cluster", vpc=vpc)

        _ = ecr.Repository(self, f"{self._SERVICE_NAME}Repo")

        task_def = ecs.FargateTaskDefinition(
            self,
            f"{self._SERVICE_NAME}TaskDef",
            cpu=256,
            memory_limit_mib=512,
        )

        _ = task_def.add_container(
            f"{self._SERVICE_NAME}Container",
            image=ecs.ContainerImage.from_asset(".", file="Dockerfile", platform=Platform.LINUX_ARM64),
            logging=ecs.LogDriver.aws_logs(
                stream_prefix=self._SERVICE_NAME,
                log_group=logs.LogGroup(self, f"{self._SERVICE_NAME}LogGroup", retention=logs.RetentionDays.ONE_WEEK),
            ),
            port_mappings=[ecs.PortMapping(container_port=8000)],
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=Duration.seconds(15),
                retries=3,
                start_period=Duration.seconds(10),
                timeout=Duration.seconds(5),
            ),
        )

        service = ecs.FargateService(
            self,
            f"{self._SERVICE_NAME}Service",
            service_name=self._SERVICE_NAME,
            cluster=cluster,
            task_definition=task_def,
            desired_count=1,
            assign_public_ip=True,
        )

        service.connections.allow_from_any_ipv4(ec2.Port.tcp(8000), "Allow HTTP access to FastAPI")


app = cdk.App()
FastApiStack(app, "FastApiStack")
app.synth()
