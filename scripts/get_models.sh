mkdir evaluationSubmission

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-in55h6cyoDu9La7duyMHnKIXQFPakyc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-in55h6cyoDu9La7duyMHnKIXQFPakyc" -O evaluationSubmission/segmentator_v1.pt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1j6jOCGkDU3kdhsFv2Vbo71O-Lt7qz8Fy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1j6jOCGkDU3kdhsFv2Vbo71O-Lt7qz8Fy" -O evaluationSubmission/segmentator_swa_v1.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14XmWq2QassuRgFk-0R0ZPmZ1MXMjACRE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14XmWq2QassuRgFk-0R0ZPmZ1MXMjACRE" -O evaluationSubmission/segmentator_weakly_v2.pt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1I6OSYwpli_q4K1ODHfOYrvUa0kRORnAL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1I6OSYwpli_q4K1ODHfOYrvUa0kRORnAL" -O evaluationSubmission/segmentator_mnmsfromweakly_v2.pt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BZFJKl26wPwWtgBVFBOj1CottC9PKIgC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BZFJKl26wPwWtgBVFBOj1CottC9PKIgC" -O evaluationSubmission/segmentator_swa_v2.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HBaqMiIeJlTePDGpaXtwgY2B6DT-WdWq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HBaqMiIeJlTePDGpaXtwgY2B6DT-WdWq" -O evaluationSubmission/segmentator_weaklyEntropyMnms_v3_1.pt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WMFbnotCzOePLUJ9bQHx8hPOnSsoOx-F' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WMFbnotCzOePLUJ9bQHx8hPOnSsoOx-F" -O evaluationSubmission/segmentator_EntropyMnms_v3_2.pt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-aTUCDyRRpIUnEVlZc6Ztsej-z7cfPfc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-aTUCDyRRpIUnEVlZc6Ztsej-z7cfPfc" -O evaluationSubmission/segmentator_EntropyMnms_swa_v3_3.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nlYrqcyM1PAUWKTlu9Fj1BYH2bS7LbKA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nlYrqcyM1PAUWKTlu9Fj1BYH2bS7LbKA" -O evaluationSubmission/discriminator_v3.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1g06rPUlNS_867gklJ0YYA5PD-zkNXGMv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1g06rPUlNS_867gklJ0YYA5PD-zkNXGMv" -O evaluationSubmission/segmentator_weakly_v5.pt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BDB5H50Hr-y3EONOWNbebs7QblrqK55a' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BDB5H50Hr-y3EONOWNbebs7QblrqK55a" -O evaluationSubmission/segmentator_v5.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nMPMwwzv-8xY0YxeHSuCW51HZJqD35-z' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nMPMwwzv-8xY0YxeHSuCW51HZJqD35-z" -O evaluationSubmission/segmentator_swa_v6.pt && rm -rf /tmp/cookies.txt
