import boto3


def get_param(
        name: str = "/yt-trending/secrets/NEPTUNE_API_KEY",
        region: str = "eu-west-1",
        with_decryption: bool = True,
) ->  str:
    # Function works only with String and SecureString parameter types.
    ssm = boto3.client("ssm", region_name=region)
    parameter = ssm.get_parameter(Name=name, WithDecryption=with_decryption)
    return parameter["Parameter"]["Value"]
