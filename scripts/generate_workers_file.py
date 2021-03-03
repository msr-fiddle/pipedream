import json
import subprocess


def print_instance_ips(instance_type):
    command = "aws ec2 describe-instances --filters Name=instance-type,Values=%s" % instance_type
    output = subprocess.check_output(command, shell=True)
    output_obj = json.loads(output)
    for i in range(len(output_obj["Reservations"])):
        instances = output_obj["Reservations"][i]["Instances"]

        for instance in instances:
            try:
                public_ip_address = instance["PublicDnsName"]
                private_ip_address = instance["PrivateIpAddress"]
                print(":".join(["ubuntu@" + public_ip_address, "22", private_ip_address]))
            except:
                pass


if __name__ == '__main__':
    print_instance_ips('p3.16xlarge')
