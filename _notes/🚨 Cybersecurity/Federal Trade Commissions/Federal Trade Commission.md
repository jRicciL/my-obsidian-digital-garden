# Cybersecurity → Federal Trade Commission

![[Pasted image 20231216131132.png]]
## 1. Cybersecurity Basics

- Cyber criminals target companies of all sizes
1. ==**Protect your files and devices**==
	1. **Update** your software
	2. **Secure your files** → Backup offline or in the cloud
	3. Require **passwords** → Passwords for all devices
	4. **Encrypt** devices → With sensitive information.
	5. Use **multi-factor authentication**
2. ==**Protect your wireless network**==
	1. Change the default name and password.
	2. Turn off remote management
	3. Use at least **WPA2** encryption → Protects information sent over your network so it can’t be read by outsiders.
3. ==**Make Smart security your business as usual**==
	1. **Require strong passwords:**
		1. At least 12 characters
		2. Never reuse passwords and don’t share them on the phone, text, or email.
		3. Limit the number of unsuccessfult log-in attemps
		4. Train all staff → Create a culture of security by implementing a regular schedule of employee training.
		5. Have a plan → For saving data, running the business and notifying costumers if you experience a breach.

## 2. Business email imposters

A scammer sets up an email address that looks like it’s from your company:
- ==Spoofing== → The scammer (business email imposter) sends out messages using the fake email adress.
	- The objective is to get passwords and bank account numbers
	- Companies has a lot to lose:
		- Customers and partners might lose trust
		- Your business may lose money
### How to protect your business
- **Use email authentication** → To validate the email address → The email comes from your companies server
- Keep your security up to date:
	- Install the lates patches and updates. 
	- Set them automatically on your network
- Train your staff:
	- Teach them how to avoid phishing scams and show them some of the common ways of attacks.
### What to do if someone spoofs your email?
1. Report it:
	- Report the scam to local law enforcement:
		- FBI’s Internet Complaint Crimes Center, and the FCT.
		- Forward phishing emails to `reportphising@apwg.org`
2. Notify your customers:
	- Tell your customers if scammers are impersonating your business.
	- Sent email without hyperlinks.
	- Remind customers not to share any personal information through email or text.
3. Alert your staff:
	- Use this experience to update your security practices and train your staff about cyber attacks.

## 3. Email Authentication
- Makes it a lot harder for a scammer to send phishing emails that look they’re from your company.
- Allows a receiving server to verify an email from your company and block emails from an imposter.
### Email Authentication tools:
- ==**Sender Policy Framework (SPF)**==
	- Contains links and attachments that put your data and network at risks.
- ==**Domain Keys Identified Mail (DKIM):**==
	- Puts a **digital signature** on outgoing mail so servers can verify that an email from your domain actually was sent from your organization’s server and hasn’t been tampered with in transit.
- **==Domain-based Message Authentication, Reporting & Conformance (DMARC):==**
	- #SPF and #DKIM verify the address the server uses “behind the scenes”
	 - #DMARC verifies that this address matches the “from” address you see.
		 - Notifies you
	 - Tells other servers what to do:
		 - Reject the email
		 - Fag it as spam
		 - Take no Action
 
## 4. Hiring a web host
### Transport Layer Security (TLS)
Capa de Seguridad de transporte
- Its predecessor is SSL → Secure Sockets Layer
- Will help to protect your customer’s privacy
- Makes sure your customers get to your real website → `https://`

### Other things to look for
- Email Authentification
	- SPF, DKIM, and DMARC
- Software updates
- Website management

### What to ask
- When you are hiring a webhost provider, ask these questions to make sure you’re helping protect your customer information and your business data
	- Is TLS included with the hosting plan?
	- Are the most up-to-date software versions available
	- Can you offer email authentification → SPF, DKIM, and DMARC


## 5. Cyber Insurance

Recovering from a cyber attack can be costly:
- Cyber insurance → 
	- Option than can help protect your business agains the results of a cyber attack.
	- First-party and third-party coverage, or both.
### What should your cyber insurance policy cover?
- Data breaches → Like theft of personal information
- Cyber attacks on your data held by vendors and other third parties.
- Cyber attacks → Like breaches of your networks
- Cyber attacks that occur anywhere in the world
- Terrorists attacks

Also consider whether your cyber insurance provider will:
- Defend you in a lawsuit or regulatory investigation
- Provide coverage in excess of any other applicable insurance you have
- Offer support 

### First party coverage
Protects your data, including employee and customer information:
- Legal counsel to determine your notification and regulatory obligations
- Recovery and replacement of lost or stolen data
- Customer notification and call center services
- Lost income due to business interruption
- Cyber extortion and fraud
- Forensic services to investigate the breach
- Fees, fines, and penalties related to the cyber incident

### Third-party coverage
- Payments to consumers affected by the breach
- Claims and settlement expenses relating to disputes or lawsuits
- Losses related to defamation and copyright or trademark infringement
- cost for litigation and responding to regulatory inquiries
- Other settlements, damages, and judgments
- Accounting costs.

## 6. The NIST Cybersecurity Framework
