## RMSE Formula
TODO 


### Read up on techonologies
 
Technologies I dont know, I need to read up on and some clarifying bullet points on each:

#### Databricks
  - Databricks is the central platform where data ingestion, feature engineering, model training, and inference jobs run.
  - Comparable tools: AWS SageMaker, Vertex AI

#### Databricks Workflows
Scheduler + orchestration system.
Databricks Workflows orchestrates the training, inference, and monitoring jobs.

Equivalent to:
- Airflow
- Tekton
- Cron jobs

#### Azure Data Lake Storage Gen2
  - Azure's big bucket for files.
  - comparable to Amazon S3

#### Parquet

#### Delta Lake
A storage format built on top of Parquet.
Adds:

ACID transactions
version history
schema enforcement
time travel

Without Delta Lake:

folder of parquet files

With Delta Lake:

folder of parquet files
+
transaction log

Delta Lake is roughly "Git for Parquet datasets": a transaction log records versions of immutable Parquet files, allowing consistent reads, history, and rollback-like time travel.

#### Delta table
A table stored in Delta Lake format.

#### Unity Catalog
Unity Catalog provides centralized governance, lineage, and access control across datasets and models.

Think:

GitHub permissions, but for data assets.

#### Databricks Asset Bundles

Infrastructure-as-code for Databricks.

Instead of manually creating jobs in the UI:

jobs:
  inference_job:
     ...

stored in Git.

Then deployed automatically.

Comparable to
Helm charts
Terraform
Kubernetes manifests
Short explanation

Asset Bundles allow Databricks jobs and resources to be deployed consistently through CI/CD pipelines. 

#### Databricks Lakehouse Monitoring

Monitoring for data quality and ML systems.

Tracks:

missing values
feature drift
prediction drift
model performance
Example

Suppose:

Average passengers:
2025 = 500
2026 = 900

Monitoring detects:

"Something changed."

Short explanation

Lakehouse Monitoring helps detect data quality issues and model drift before prediction quality deteriorates significantly.

#### Azure Monitor

Azure-wide monitoring platform.

Collects:

logs
metrics
alerts

from:

Databricks
Azure Functions
VMs
Databases
Comparable to
Prometheus + Grafana
CloudWatch
Short explanation

Azure Monitor provides centralized operational monitoring across all Azure services involved in the solution.

#### Databricks alerts

Simple alerting mechanism.

Example:

RMSE > 50

Send:

email
Slack
Teams
Short explanation

Databricks Alerts notify operators when predefined thresholds are exceeded. 

#### Azure Function

Serverless code execution.

You upload:

def handler():
    ...

Azure runs it when triggered.

Good for
lightweight APIs
webhooks
event handling
Bad for
Spark jobs
ML training
pandas-heavy workloads
Why not use it here?

Your workload is:

scheduled
batch-oriented
ML-focused

Databricks is a much better fit.

Short explanation

Azure Functions are useful for lightweight event-driven processing, but model training and forecasting are better handled by Databricks jobs.