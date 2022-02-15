import paramiko
import sys
import os
import time

# def deleteRemoteFile(dt):
#     ssh = paramiko.SSHClient()
#     ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())#第一次登录的认证信息
#     ssh.connect(hostname='34.80.123.50', port=22, username='root', password='19990712@Fu') # 连接服务器
#     stdin, stdout, stderr = ssh.exec_command('rm /home/gold/data//*') # 执行命令
#     ssh.close()

def uploadFile2Remote():
    transport = paramiko.Transport(('34.80.123.50', 22))
    transport.connect(username='root', password='19990712@Fu')
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put('/home/fla/workspace/preprocessing/log.txt', '/root/temp_dl/log.txt')
    transport.close()
    print("Send! {}".format(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())))
 
if __name__ == '__main__':
    while 1:
        uploadFile2Remote()
        time.sleep(3600)
