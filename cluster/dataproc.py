import logging

import time
from googleapiclient.discovery import build
from google.oauth2 import service_account
import argparse

PROJECT_ID = 'google_cloud_progect_id'
REGION = 'us-east1'
ZONE = 'us-east1-b'
MAP_BUILDER_USER = 'userok'
GOOGLE_CLOUD_APP_CREDENTIALS = './cred.json'
logger = logging.getLogger(__name__)

class DataProcError(Exception):
    pass

def get_resource():
    credentials = service_account.Credentials.from_service_account_file(
        GOOGLE_CLOUD_APP_CREDENTIALS)
    scoped_credentials = credentials.with_scopes([
        'https://www.googleapis.com/auth/cloud-platform'
    ])
    return build('dataproc', 'v1', credentials=scoped_credentials,
                 cache_discovery=False)


def create_cluster(dataproc, cluster_name, num_workers):
    zone_uri = (
        'https://www.googleapis.com/compute/v1/projects/{}/zones/{}'
        .format(PROJECT_ID, ZONE)
    )
    cluster_data = {
        'projectId': PROJECT_ID,
        'clusterName': cluster_name,
        'config': {
            'gceClusterConfig': {
                'zoneUri': zone_uri
            },
            'masterConfig': {
                'diskConfig': {
                    'bootDiskSizeGb': 1024
                },
                'machineTypeUri': 'n1-highmem-16'
            },
            'workerConfig': {
                'numInstances': num_workers,
                'machineTypeUri': 'n1-highmem-16'
            }
        }
    }
    logger.info('Creating %s cluster', cluster_name)
    dataproc.projects().regions().clusters().create(
        projectId=PROJECT_ID, region=REGION, body=cluster_data
    ).execute()
    state = 'CREATING'
    config = None
    while state == 'CREATING':
        time.sleep(10)
        config = cluster_config(dataproc, cluster_name)
        if not config:
            raise DataProcError('No cluster "{}" running'.format(cluster_name))
        state = config['status']['state']
    if state != 'RUNNING':
        raise DataProcError('Dataproc cluster was created with status {}'
                            .format(state))
    logger.info('Cluster %s was set up', cluster_name)
    return config


def cluster_config(dataproc, cluster_name):
    clusters = dataproc.projects().regions().clusters().list(
        projectId=PROJECT_ID, region=REGION
    ).execute()
    cluster = [c for c in clusters['clusters']
               if c['clusterName'] == cluster_name]
    if not cluster:
        raise DataProcError('No clusters running')
    cluster = cluster[0]
    return cluster


def terminate_cluster(dataproc, cluster_name):
    logger.info('Terminating %s cluster', cluster_name)
    result = dataproc.projects().regions().clusters().delete(
        projectId=PROJECT_ID, region=REGION, clusterName=cluster_name
    ).execute()
    logger.info('Cluster %s was terminated', cluster_name)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="sparktf",
        help="cluster name"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=2,
        help="number of workers"
    )
    parser.add_argument(
        "action",
        type=str,
        help="deploy or delete"
    )
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.action == "deploy":
        dataproc = get_resource()
        create_cluster(dataproc, FLAGS.name, FLAGS.n)    
    elif FLAGS.action == "delete":
        dataproc = get_resource()
        terminate_cluster(dataproc, FLAGS.name,)
    else:
        print("wrong action")

