import time


def time_circle():
    import os
    import subprocess

    # os.system只会输出 得不到结果
    # ans = os.system("cd /home/lenong0427 ; ls -l")
    # print(ans)

    # os.popen 可以得到输出的结果 多条命令用;分隔
    will_search = ['teamviewerd','todeskd',"zerotier-one"]
    will_do = ['teamviewer','sudo systemctl restart todeskd.service','sudo zerotier-one -d']
    import pexpect

    for (i,will) in enumerate(will_search):

        # lines = os.popen(f"ps -e | grep {will}").readlines()
        # print(len(lines))
        #
        # if will_search[i].startswith("todeskd"):
        #     child = pexpect.spawn(will_do[i])
        #     child.expect("sudo")
        #     child.sendline("23456")
        #     print("success")
        #     time.sleep(1)
        #     os.system("sudo todesk")

        if will_search[i].startswith("zerotier-one"):
            print(will_search[i])

            p = " ps -e | grep zerotier-one"
            p = os.popen(p).readlines()
            print(p)
            if len(p) != 0:
                p = p[0]
                p = p.split("?")[0].strip()
                print(p)
                child = pexpect.spawn(f"sudo kill -9 {p}")
                child.expect("sudo")
                child.sendline("23456")
                time.sleep(10)
                child = pexpect.spawn(will_do[i])
                child.expect("sudo")
                child.sendline("23456")

                for ls in child.readlines():
                    ls = str(ls,encoding='utf-8')
                    print(ls)
            else:
                print(0)
                child = pexpect.spawn(will_do[i])
                child.expect("sudo")
                child.sendline("23456")
                for ls in child.readlines():
                    ls = str(ls,encoding='utf-8')
                    print(ls)





if __name__ == '__main__':
    time_circle()