import os
from typing import Any

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    Stack,
    aws_ec2,
    aws_ecs,
    aws_ecs_patterns,
    aws_elasticloadbalancingv2,
    aws_logs,
    aws_route53,
)
from aws_cdk.aws_ecr_assets import Platform
from constructs import Construct


class FastApiStack(Stack):
    _SERVICE_NAME: str = "SWEBench"

    def __init__(self, scope: Construct, id: str, **kwargs: Any):
        super().__init__(scope, id, **kwargs)

        # don't create a NAT Gateway
        vpc = aws_ec2.Vpc(
            self,
            f"{self._SERVICE_NAME}Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                aws_ec2.SubnetConfiguration(
                    name="Public", subnet_type=aws_ec2.SubnetType.PUBLIC, map_public_ip_on_launch=True
                )
            ],
        )

        cluster = aws_ecs.Cluster(self, f"{self._SERVICE_NAME}Cluster", vpc=vpc)

        # get our domain zone
        hosted_zone = aws_route53.HostedZone.from_lookup(self, "HostedZone", domain_name="vals.ai")

        # fargate task (no docker in docker support!)
        task_def = aws_ecs.FargateTaskDefinition(
            self,
            f"{self._SERVICE_NAME}TaskDef",
            cpu=2048,  # 2 vCPUs
            memory_limit_mib=4096,  # 4 GB RAM
            runtime_platform=aws_ecs.RuntimePlatform(
                cpu_architecture=aws_ecs.CpuArchitecture.X86_64,
                operating_system_family=aws_ecs.OperatingSystemFamily.LINUX,
            ),
        )

        self.container = task_def.add_container(
            f"{self._SERVICE_NAME}Container",
            image=aws_ecs.ContainerImage.from_asset(".", file="Dockerfile", platform=Platform.LINUX_AMD64),
            # logs - 1 week retention
            logging=aws_ecs.LogDriver.aws_logs(
                stream_prefix=self._SERVICE_NAME,
                log_group=aws_logs.LogGroup(
                    self,
                    f"{self._SERVICE_NAME}LogGroup",
                    retention=aws_logs.RetentionDays.ONE_WEEK,
                    removal_policy=cdk.RemovalPolicy.DESTROY,
                ),
            ),
            port_mappings=[aws_ecs.PortMapping(container_port=8000)],
            # container health check - every 30s
            health_check=aws_ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=Duration.seconds(30),
                retries=3,
                start_period=Duration.seconds(15),
                timeout=Duration.seconds(5),
            ),
        )

        # load balanced with public domain
        self.service = aws_ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            f"{self._SERVICE_NAME}Service",
            cluster=cluster,
            desired_count=1,
            task_definition=task_def,
            service_name=self._SERVICE_NAME,
            circuit_breaker=aws_ecs.DeploymentCircuitBreaker(
                rollback=True
            ),  # rollback to previous version if task fails
            domain_name="benchmark-service.vals.ai",
            domain_zone=hosted_zone,
            protocol=aws_elasticloadbalancingv2.ApplicationProtocol.HTTPS,
            redirect_http=True,
            open_listener=False,  # security group not configured, need to manually add whitelisted IPs (no public access)
            assign_public_ip=True,
            public_load_balancer=True,
        )

        # load balancer health check - every 20s
        self.service.target_group.configure_health_check(path="/health", port="8000", interval=Duration.seconds(20))

        # set request timeout - 1min
        self.service.load_balancer.set_attribute("idle_timeout.timeout_seconds", "60")

        # autoscaling - max 2 tasks, scale when CPU > 80%
        scaling = self.service.service.auto_scale_task_count(min_capacity=1, max_capacity=2)
        scaling.scale_on_cpu_utilization("CpuScaling", target_utilization_percent=70)


app = cdk.App()
FastApiStack(
    app,
    "FastApiStack",
    env=cdk.Environment(account=os.getenv("CDK_DEFAULT_ACCOUNT"), region=os.getenv("CDK_DEFAULT_REGION")),
)
app.synth()
