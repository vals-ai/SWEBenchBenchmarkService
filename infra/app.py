from typing import Any

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    Stack,
    aws_ec2,
    aws_ecr,
    aws_ecs,
    aws_logs,
)
from aws_cdk.aws_ecr_assets import Platform
from constructs import Construct


class FastApiStack(Stack):
    _SERVICE_NAME: str = "Swebench"

    def __init__(self, scope: Construct, id: str, **kwargs: Any):
        super().__init__(scope, id, **kwargs)

        vpc = aws_ec2.Vpc(self, f"{self._SERVICE_NAME}Vpc", max_azs=2)

        cluster = aws_ecs.Cluster(self, f"{self._SERVICE_NAME}Cluster", vpc=vpc)

        _ = aws_ecr.Repository(self, f"{self._SERVICE_NAME}Repo")

        task_def = aws_ecs.FargateTaskDefinition(
            self,
            f"{self._SERVICE_NAME}TaskDef",
            cpu=1024,
            memory_limit_mib=2048,
        )

        _ = task_def.add_container(
            f"{self._SERVICE_NAME}Container",
            image=aws_ecs.ContainerImage.from_asset(".", file="Dockerfile", platform=Platform.LINUX_ARM64),
            logging=aws_ecs.LogDriver.aws_logs(
                stream_prefix=self._SERVICE_NAME,
                log_group=aws_logs.LogGroup(
                    self, f"{self._SERVICE_NAME}LogGroup", retention=aws_logs.RetentionDays.ONE_WEEK
                ),
            ),
            port_mappings=[aws_ecs.PortMapping(container_port=8000)],
            health_check=aws_ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=Duration.minutes(5),
                retries=3,
                start_period=Duration.seconds(10),
                timeout=Duration.seconds(5),
            ),
        )

        service = aws_ecs.FargateService(
            self,
            f"{self._SERVICE_NAME}Service",
            service_name=self._SERVICE_NAME,
            cluster=cluster,
            task_definition=task_def,
            desired_count=1,
            assign_public_ip=True,
        )

        service.connections.allow_from_any_ipv4(
            port_range=aws_ec2.Port.tcp(8000), description="Allow HTTP access to FastAPI"
        )


app = cdk.App()
FastApiStack(app, "FastApiStack")
app.synth()
