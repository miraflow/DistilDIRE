#!/bin/bash

curl --output /media/changyeon/DIRE/datasets/image/imagenet_image_real_train.tar.gz 'https://recstore.ustc.edu.cn/file/20230816_ecb531f2cc8363998a3841b5d37e3e32?Signature=7+doX+sAgcOjwd5av2zr4D7C5F0=&Expires=1711330312&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22real.tar.gz%22&storage=moss&filename=real.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: document' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/imagenet_image_fake_train.tar.gz 'https://recstore.ustc.edu.cn/file/20230816_7f0cb233db2fcd8df635806d40fc6d4b?Signature=mkKEA/Kl4B1W3kx80h2lhpve2uc=&Expires=1711330283&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22adm.tar.gz%22&storage=moss&filename=adm.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: document' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/imagenet_image_real_val.tar.gz 'https://recstore.ustc.edu.cn/file/20230815_2f97ad6ee6683818c36e56f27ee4495c?Signature=DFqU3VuI0wj9M/xypds0Ofs6i/Q=&Expires=1711330339&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22real.tar.gz%22&storage=moss&filename=real.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: iframe' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/imagenet_image_fake_val.tar.gz 'https://recstore.ustc.edu.cn/file/20230815_7117dfe81c260dd7d408c40811511239?Signature=hkheElBxkTgkiFAEIvVFQPZrZHg=&Expires=1711330384&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22adm.tar.gz%22&storage=moss&filename=adm.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: iframe' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/imagenet_image_real_test.tar.gz 'https://recstore.ustc.edu.cn/file/20230815_dede86cc6c31ed7b528e5d438bdafca8?Signature=8dXMX7ab8Sg9M2cJdPVKc7me9qc=&Expires=1711330417&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22real.tar.gz%22&storage=moss&filename=real.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: iframe' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/imagenet_image_fake_adm_test.tar.gz 'https://recstore.ustc.edu.cn/file/20230815_d1967940171791ac71e190ae96e5df28?Signature=9lNzlgUt+02f/QOkjkrsqGzhoWo=&Expires=1711330478&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22adm.tar.gz%22&storage=moss&filename=adm.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: iframe' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/imagenet_image_fake_sdv_test.tar.gz 'https://recstore.ustc.edu.cn/file/20230815_cf127ad3383d87d7a04bd78d9b14d849?Signature=Cnd+Npum/hI3r4lxnzjUFFO/Jw4=&Expires=1711330502&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22sdv1.tar.gz%22&storage=moss&filename=sdv1.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: iframe' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/celebahq_image_real_train.tar.gz 'https://recstore.ustc.edu.cn/file/20230816_51c74fe2b29c2604e9f5ff6a86a114c2?Signature=/dDCIcqGdHDLjXEcI6v9b08BKH8=&Expires=1711330155&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22real.tar.gz%22&storage=moss&filename=real.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: document' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/celebahq_image_fake_train.tar.gz 'https://recstore.ustc.edu.cn/file/20230816_6a30bcbaeb6906cbc65fb4c71bcb8577?Signature=CnPeJmQMiw4RjQOzZ/tV0/aQJN0=&Expires=1711330110&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22sdv2.tar.gz%22&storage=moss&filename=sdv2.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: document' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

# curl --output /media/changyeon/DIRE/celebahq_real_val.tar.gz
# curl --output /media/changyeon/DIRE/celebahq_fake_val.tar.gz

curl --output /media/changyeon/DIRE/datasets/image/celebahq_image_real_test.tar.gz 'https://recstore.ustc.edu.cn/file/20230815_fc19fec697a20a1e6c88297ca6a90948?Signature=tyPbwoIiLyUdrMIf5XUlcRvGmso=&Expires=1711330055&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22real.tar.gz%22&storage=moss&filename=real.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: document' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/celebahq_image_fake_dalle2_test.tar.gz 'https://recstore.ustc.edu.cn/file/20230815_26afc6ad2a83a391099c84dfab1275f8?Signature=a/ohgFWueDodPU2LiHeab8Ld92U=&Expires=1711329773&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22dalle2.tar.gz%22&storage=moss&filename=dalle2.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: iframe' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/celebahq_image_fake_sdv2_test.tar.gz 'https://recstore.ustc.edu.cn/file/20230815_8d694526b7890c084333895d4c2966db?Signature=MNmlliCRKFAKB8wS3I5lLoR1tvY=&Expires=1711329796&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22sdv2.tar.gz%22&storage=moss&filename=sdv2.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: iframe' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/celebahq_image_fake_midjourney_test.tar.gz 'https://recstore.ustc.edu.cn/file/20230815_0243cbb13311cd9570989eff6c383780?Signature=4phZZR8nwAJtEPpsfOmSXUAb5yo=&Expires=1711329823&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22midjourney.tar.gz%22&storage=moss&filename=midjourney.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: iframe' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'

curl --output /media/changyeon/DIRE/datasets/image/celebahq_image_fake_if_test.tar.gz 'https://recstore.ustc.edu.cn/file/20230815_f76d31664a0adf6ea704a5ed5b5d6b98?Signature=xC4b0+VU8LnlkkKGmBIfaCq4XwM=&Expires=1711329884&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22if.tar.gz%22&storage=moss&filename=if.tar.gz&download=download' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7' \
  -H 'Connection: keep-alive' \
  -H 'Referer: https://rec.ustc.edu.cn/' \
  -H 'Sec-Fetch-Dest: iframe' \
  -H 'Sec-Fetch-Mode: navigate' \
  -H 'Sec-Fetch-Site: same-site' \
  -H 'Sec-Fetch-User: ?1' \
  -H 'Upgrade-Insecure-Requests: 1' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"'