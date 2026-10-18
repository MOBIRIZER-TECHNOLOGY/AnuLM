"""
SYNTHETIC Indian-banking intents for the Decide demo: 20 intents customers of Indian banks
raise (UPI, IFSC, KYC, Aadhaar, EMI, ...), each in English, Hindi and Hinglish.

    python experiments/make_india_banking.py     # -> data/decide_general/india_banking/{train,test}.jsonl

Every sentence here was written for this demo, not collected from customers: it shows the
model can learn an Indian bank's own intent list, and its score is only as good as these
templates. The test set uses phrasings (templates) that never appear in train, so it
measures generalisation to new wordings, not to real traffic. Replace with a client's real
labelled queries for anything beyond a demo.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "decide_general" / "india_banking"

APPS = ["PhonePe", "Google Pay", "GPay", "Paytm", "BHIM", "Amazon Pay"]
BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "Bank of Baroda", "Canara Bank"]
AMTS = ["500", "1,200", "2000", "3,500", "5000", "10,000", "15000", "25,000"]
DAYS = ["2", "3", "4", "5", "7"]

# intent: {lang: [templates]}; the LAST TWO of each list are test-only phrasings.
T = {
    "upi_payment_failed": {
        "en": ["My UPI payment of Rs {amt} failed", "UPI transaction failed on {app}, please help",
               "Payment through {app} is not going through", "Why does my UPI payment keep failing?",
               "I tried paying {amt} by UPI and it showed failed", "UPI is not working for me today",
               "Transaction declined while paying via {app}",
               "Every time I scan the QR code the UPI payment fails", "{app} says payment unsuccessful"],
        "hi": ["मेरा यूपीआई पेमेंट फेल हो गया", "{app} से पेमेंट नहीं हो रहा है", "यूपीआई से {amt} रुपये भेजने में दिक्कत आ रही है",
               "यूपीआई ट्रांजैक्शन बार बार फेल हो रहा है", "पेमेंट करते समय फेल्ड दिखा रहा है",
               "क्यूआर कोड स्कैन करने पर भुगतान नहीं होता", "यूपीआई काम नहीं कर रहा, क्या करूं"],
        "hinglish": ["mera upi payment fail ho gaya", "{app} se payment nahi ho raha", "upi transaction baar baar fail ho raha hai",
                     "{amt} bhejne ki koshish ki par failed dikha raha", "upi kaam nahi kar raha aaj",
                     "QR scan karke pay kiya to fail ho gaya", "payment unsuccessful bol raha hai {app} pe"],
    },
    "money_debited_not_credited": {
        "en": ["Money was debited from my account but the receiver did not get it",
               "Rs {amt} deducted but transaction failed", "Amount debited but not credited to beneficiary",
               "My account shows debit of {amt} but the shop says they didn't receive it",
               "Failed transaction but money got cut from my account", "When will I get back the money debited for a failed payment?",
               "{amt} rupees gone from my account, recipient says nothing came",
               "Paid via {app}, money left my account, merchant has not received it",
               "Transaction failed yet my balance reduced by {amt}"],
        "hi": ["खाते से पैसे कट गए लेकिन सामने वाले को नहीं मिले", "{amt} रुपये कट गए पर ट्रांजैक्शन फेल हो गया",
               "पैसे डेबिट हो गए लेकिन क्रेडिट नहीं हुए", "फेल हुए पेमेंट का पैसा वापस कब आएगा",
               "दुकानदार कह रहा है पैसा नहीं आया पर मेरे खाते से कट गया",
               "मेरे बैलेंस से {amt} कम हो गए पर लेन देन असफल रहा", "भुगतान विफल हुआ फिर भी राशि कट गई"],
        "hinglish": ["paise kat gaye par samne wale ko nahi mile", "{amt} debit ho gaye lekin transaction fail",
                     "account se paisa cut gaya, credit nahi hua", "failed payment ka refund kab aayega",
                     "shopkeeper bol raha paisa nahi aaya, mera kat gaya",
                     "{app} se pay kiya, paisa gaya par merchant ko nahi mila", "balance kam ho gaya par payment fail dikha raha"],
    },
    "upi_pin_reset": {
        "en": ["How do I reset my UPI PIN?", "I forgot my UPI PIN", "Change UPI PIN on {app}",
               "UPI PIN is blocked after wrong attempts", "Set a new UPI PIN for my {bank} account",
               "How to create UPI PIN with debit card", "My UPI PIN is not accepted",
               "I need to generate a fresh UPI PIN", "Entered wrong UPI PIN three times, now what?"],
        "hi": ["यूपीआई पिन कैसे बदलें", "मैं अपना यूपीआई पिन भूल गया", "यूपीआई पिन गलत डालने से ब्लॉक हो गया",
               "{bank} खाते के लिए नया यूपीआई पिन सेट करना है", "डेबिट कार्ड से यूपीआई पिन कैसे बनाएं",
               "मुझे नया यूपीआई पिन बनाना है", "तीन बार गलत पिन डाला अब क्या करूं"],
        "hinglish": ["upi pin kaise reset kare", "main apna upi pin bhool gaya", "upi pin galat dalne se block ho gaya",
                     "{app} me upi pin change karna hai", "debit card se upi pin kaise banaye",
                     "naya upi pin generate karna hai", "teen baar wrong pin daala, ab kya karu"],
    },
    "neft_imps_transfer": {
        "en": ["How do I send money by NEFT?", "Transfer {amt} via IMPS to another bank",
               "What is the NEFT limit per day?", "RTGS transfer to {bank} account", "How long does NEFT take?",
               "Add a beneficiary for IMPS transfer", "Is IMPS available on Sunday?",
               "I want to move {amt} to my brother's {bank} account by NEFT",
               "Which is faster for a bank transfer, NEFT or IMPS?"],
        "hi": ["एनईएफटी से पैसे कैसे भेजें", "आईएमपीएस से {amt} रुपये दूसरे बैंक में भेजने हैं", "एनईएफटी की रोज की लिमिट क्या है",
               "{bank} खाते में आरटीजीएस करना है", "बेनिफिशियरी कैसे जोड़ें",
               "भाई के {bank} खाते में {amt} ट्रांसफर करने हैं", "एनईएफटी और आईएमपीएस में कौन सा जल्दी पहुंचता है"],
        "hinglish": ["neft se paise kaise bheje", "imps se {amt} dusre bank me bhejna hai", "neft ki daily limit kya hai",
                     "{bank} account me rtgs karna hai", "beneficiary kaise add kare",
                     "bhai ke {bank} account me {amt} transfer karne hai", "neft fast hai ya imps"],
    },
    "ifsc_code": {
        "en": ["What is the IFSC code of my branch?", "Where can I find my IFSC code?", "IFSC code for {bank} branch",
               "I need the IFSC to receive money", "Is the IFSC printed on the cheque book?",
               "My IFSC code changed after the merger", "Give me the branch IFSC",
               "Tell me the IFSC of my home branch", "Which IFSC should I share for a transfer?"],
        "hi": ["मेरी ब्रांच का आईएफएससी कोड क्या है", "आईएफएससी कोड कहां मिलेगा", "{bank} ब्रांच का आईएफएससी बताइए",
               "पैसे मंगवाने के लिए आईएफएससी चाहिए", "विलय के बाद आईएफएससी बदल गया क्या",
               "मेरी होम ब्रांच का आईएफएससी क्या है", "ट्रांसफर के लिए कौन सा आईएफएससी दूं"],
        "hinglish": ["meri branch ka ifsc code kya hai", "ifsc code kaha milega", "{bank} branch ka ifsc batao",
                     "paise mangwane ke liye ifsc chahiye", "merger ke baad ifsc badal gaya kya",
                     "home branch ka ifsc kya hai", "transfer ke liye kaunsa ifsc du"],
    },
    "kyc_update": {
        "en": ["How do I update my KYC?", "My account is frozen due to pending KYC", "Re-KYC reminder, what documents are needed?",
               "Can I do video KYC?", "Update KYC online for my {bank} account", "KYC expired, account blocked",
               "Submit address proof for KYC",
               "Bank message says complete KYC or account will be restricted", "Which documents count as valid KYC?"],
        "hi": ["केवाईसी कैसे अपडेट करें", "केवाईसी न होने से खाता बंद हो गया", "री केवाईसी के लिए कौन से दस्तावेज चाहिए",
               "क्या वीडियो केवाईसी हो सकती है", "{bank} खाते की केवाईसी ऑनलाइन करनी है",
               "बैंक का मैसेज आया है कि केवाईसी पूरी करें", "केवाईसी के लिए कौन से कागज मान्य हैं"],
        "hinglish": ["kyc kaise update kare", "kyc pending hone se account freeze ho gaya", "re kyc ke liye kya documents chahiye",
                     "video kyc ho sakti hai kya", "{bank} account ki kyc online karni hai",
                     "bank ka msg aaya kyc complete karo", "kyc ke liye kaunse papers valid hai"],
    },
    "aadhaar_link": {
        "en": ["How do I link Aadhaar to my bank account?", "Seed Aadhaar with my {bank} account",
               "Is my Aadhaar linked to my account?", "Aadhaar linking failed", "Link Aadhaar for DBT subsidy",
               "Update Aadhaar number in bank records", "Aadhaar not linked so subsidy not coming",
               "Check whether Aadhaar seeding is done", "Gas subsidy stopped, need Aadhaar linked to account"],
        "hi": ["आधार को बैंक खाते से कैसे जोड़ें", "{bank} खाते में आधार लिंक करना है", "क्या मेरा आधार खाते से जुड़ा है",
               "आधार लिंक नहीं हो रहा", "सब्सिडी के लिए आधार सीडिंग करनी है",
               "आधार सीडिंग हुई या नहीं कैसे देखें", "गैस सब्सिडी रुक गई, आधार खाते से जोड़ना है"],
        "hinglish": ["aadhaar ko bank account se kaise link kare", "{bank} account me aadhaar link karna hai",
                     "kya mera aadhaar linked hai", "aadhaar linking fail ho rahi hai", "subsidy ke liye aadhaar seeding karni hai",
                     "aadhaar seeding hui ki nahi kaise check kare", "gas subsidy band ho gayi, aadhaar link karna hai"],
    },
    "pan_link": {
        "en": ["How to update PAN in my bank account?", "Link PAN card to account", "Bank is asking for PAN, where do I submit it?",
               "TDS deducted because PAN not updated", "Add PAN to my {bank} account", "Form 60 instead of PAN",
               "PAN details wrong in account",
               "Interest got higher TDS since my PAN isn't on record", "Where do I upload my PAN card?"],
        "hi": ["बैंक खाते में पैन कैसे अपडेट करें", "पैन कार्ड खाते से लिंक करना है", "पैन अपडेट न होने से टीडीएस कट गया",
               "{bank} खाते में पैन जोड़ना है", "पैन की जानकारी गलत है",
               "पैन रिकॉर्ड में नहीं है इसलिए ज्यादा टीडीएस कटा", "पैन कार्ड कहां अपलोड करूं"],
        "hinglish": ["bank account me pan kaise update kare", "pan card account se link karna hai", "pan update nahi tha to tds kat gaya",
                     "{bank} account me pan add karna hai", "pan details galat hai account me",
                     "pan record me nahi hai isliye zyada tds kata", "pan card kaha upload karu"],
    },
    "card_block": {
        "en": ["Block my debit card immediately", "My card is lost, please block it", "Someone stole my wallet with my ATM card",
               "Hotlist my credit card", "I see transactions I didn't do, block the card", "Freeze my card for now",
               "How do I block my card by SMS?",
               "Lost my purse, my {bank} card was inside", "Unknown person used my card, stop it now"],
        "hi": ["मेरा डेबिट कार्ड तुरंत ब्लॉक करें", "कार्ड खो गया है ब्लॉक कर दो", "किसी ने मेरा एटीएम कार्ड चुरा लिया",
               "मेरे कार्ड से अनजान लेन देन हो रहे हैं", "एसएमएस से कार्ड कैसे ब्लॉक करें",
               "पर्स खो गया उसमें {bank} का कार्ड था", "किसी और ने मेरा कार्ड इस्तेमाल किया, अभी रोकिए"],
        "hinglish": ["mera debit card turant block karo", "card kho gaya hai block kar do", "kisi ne mera atm card chura liya",
                     "card se unknown transactions ho rahe hai", "sms se card kaise block kare",
                     "purse kho gaya usme {bank} ka card tha", "kisi aur ne card use kiya, abhi band karo"],
    },
    "atm_cash_not_dispensed": {
        "en": ["ATM did not give cash but money was deducted", "Cash not dispensed, account debited {amt}",
               "ATM transaction failed and amount deducted", "Withdrew {amt} at ATM, no cash came out",
               "Other bank ATM didn't dispense money", "How long for ATM failed transaction refund?",
               "ATM gave less money than I entered",
               "Machine showed transaction complete but no notes came", "Money cut at {bank} ATM without cash"],
        "hi": ["एटीएम से पैसे नहीं निकले पर खाते से कट गए", "एटीएम ने कैश नहीं दिया और {amt} कट गए",
               "दूसरे बैंक के एटीएम से पैसे नहीं निकले", "एटीएम फेल ट्रांजैक्शन का पैसा कब लौटेगा", "एटीएम ने कम पैसे दिए",
               "मशीन ने लेन देन पूरा दिखाया पर नोट नहीं निकले", "{bank} एटीएम पर बिना कैश के पैसे कट गए"],
        "hinglish": ["atm se paise nahi nikle par account se kat gaye", "atm ne cash nahi diya aur {amt} kat gaye",
                     "dusre bank ke atm se paise nahi nikle", "atm failed transaction ka refund kab aayega", "atm ne kam paise diye",
                     "machine ne transaction complete dikhaya par note nahi aaye", "{bank} atm pe bina cash ke paisa cut gaya"],
    },
    "loan_emi": {
        "en": ["When is my loan EMI due?", "Can I prepay my home loan?", "EMI bounced, what are the charges?",
               "Change EMI date for my personal loan", "How much EMI is left on my car loan?", "Convert purchase to EMI",
               "Loan EMI got deducted twice",
               "I want to foreclose my loan early", "How many EMIs are pending on my {bank} loan?"],
        "hi": ["मेरी लोन की ईएमआई कब कटेगी", "होम लोन का प्रीपेमेंट कर सकते हैं क्या", "ईएमआई बाउंस हो गई, कितना चार्ज लगेगा",
               "पर्सनल लोन की ईएमआई की तारीख बदलनी है", "लोन की ईएमआई दो बार कट गई",
               "लोन जल्दी बंद करना चाहता हूं", "{bank} लोन की कितनी ईएमआई बाकी हैं"],
        "hinglish": ["meri loan emi kab kategi", "home loan prepay kar sakte hai kya", "emi bounce ho gayi, kitna charge lagega",
                     "personal loan ki emi date change karni hai", "loan emi do baar kat gayi",
                     "loan jaldi close karna hai", "{bank} loan ki kitni emi baaki hai"],
    },
    "credit_card_bill": {
        "en": ["What is my credit card bill amount?", "Pay credit card bill through {app}", "Credit card due date",
               "Minimum amount due on my card", "Late payment fee charged on credit card", "Credit card statement not received",
               "Paid my card bill but it still shows due",
               "How much do I owe on my {bank} credit card this month?", "Can I pay only the minimum due?"],
        "hi": ["क्रेडिट कार्ड का बिल कितना है", "{app} से क्रेडिट कार्ड बिल भरना है", "क्रेडिट कार्ड की ड्यू डेट क्या है",
               "क्रेडिट कार्ड पर लेट फीस लग गई", "बिल भर दिया फिर भी बकाया दिखा रहा है",
               "इस महीने {bank} क्रेडिट कार्ड का कितना देना है", "क्या सिर्फ मिनिमम ड्यू भर सकता हूं"],
        "hinglish": ["credit card ka bill kitna hai", "{app} se credit card bill pay karna hai", "credit card ki due date kya hai",
                     "credit card pe late fee lag gayi", "bill bhar diya phir bhi due dikha raha",
                     "is mahine {bank} card ka kitna dena hai", "sirf minimum due bhar sakta hu kya"],
    },
    "account_balance": {
        "en": ["What is my account balance?", "Check balance in my savings account", "How much money is in my {bank} account?",
               "Balance enquiry", "Tell me my available balance", "Missed call number for balance",
               "Show me my current balance",
               "How much do I have left in my account?", "Is my salary credited, what's the balance now?"],
        "hi": ["मेरे खाते में कितना बैलेंस है", "सेविंग खाते का बैलेंस चेक करना है", "{bank} खाते में कितने पैसे हैं",
               "बैलेंस जानने का मिस्ड कॉल नंबर क्या है", "मेरा उपलब्ध बैलेंस बताइए",
               "खाते में कितने पैसे बचे हैं", "सैलरी आई क्या, अभी बैलेंस कितना है"],
        "hinglish": ["mere account me kitna balance hai", "savings account ka balance check karna hai",
                     "{bank} account me kitne paise hai", "balance ke liye missed call number kya hai", "available balance batao",
                     "account me kitne paise bache hai", "salary aayi kya, abhi balance kitna hai"],
    },
    "mini_statement": {
        "en": ["Send me a mini statement", "Last five transactions", "I need my bank statement for 6 months",
               "Download account statement PDF", "Statement for loan application", "Email my passbook statement",
               "Show recent transactions on my account",
               "Need statement of last 3 months for visa", "Update my passbook entries"],
        "hi": ["मिनी स्टेटमेंट भेजिए", "पिछले पांच लेन देन दिखाइए", "छह महीने का बैंक स्टेटमेंट चाहिए",
               "खाते का स्टेटमेंट पीडीएफ डाउनलोड करना है", "लोन के लिए स्टेटमेंट चाहिए",
               "वीजा के लिए पिछले तीन महीने का स्टेटमेंट चाहिए", "पासबुक अपडेट करवानी है"],
        "hinglish": ["mini statement bhejo", "last paanch transactions dikhao", "6 mahine ka bank statement chahiye",
                     "account statement pdf download karna hai", "loan ke liye statement chahiye",
                     "visa ke liye last 3 mahine ka statement chahiye", "passbook update karwani hai"],
    },
    "cheque_book": {
        "en": ["Request a new cheque book", "How do I order a chequebook?", "Stop payment on a cheque",
               "Cheque book not delivered yet", "My cheque bounced, what now?", "Cancelled cheque for EMI mandate",
               "How many leaves in a cheque book?",
               "Please send a 25 leaf cheque book to my address", "I issued a cheque by mistake, stop it"],
        "hi": ["नई चेक बुक चाहिए", "चेक बुक कैसे मंगवाएं", "चेक का पेमेंट रोकना है", "चेक बुक अभी तक नहीं आई",
               "मेरा चेक बाउंस हो गया", "मेरे पते पर पच्चीस पन्नों वाली चेक बुक भेजिए", "गलती से चेक दे दिया, रोक दीजिए"],
        "hinglish": ["nayi cheque book chahiye", "cheque book kaise order kare", "cheque ka payment stop karna hai",
                     "cheque book abhi tak nahi aayi", "mera cheque bounce ho gaya",
                     "mere address pe 25 page ki cheque book bhejo", "galti se cheque de diya, stop karo"],
    },
    "fixed_deposit": {
        "en": ["Open a fixed deposit of Rs {amt}", "What are the FD interest rates?", "Break my FD before maturity",
               "FD interest for senior citizens", "Auto renew my fixed deposit", "Loan against FD",
               "When does my FD mature?",
               "I want to invest {amt} for one year in an FD", "Is there a penalty if I close my FD early?"],
        "hi": ["{amt} रुपये की एफडी खोलनी है", "एफडी पर ब्याज दर क्या है", "एफडी समय से पहले तोड़नी है",
               "सीनियर सिटीजन को एफडी पर कितना ब्याज", "मेरी एफडी कब मैच्योर होगी",
               "एक साल के लिए {amt} की एफडी करनी है", "एफडी जल्दी बंद करने पर पेनल्टी लगेगी क्या"],
        "hinglish": ["{amt} ki fd kholni hai", "fd pe interest rate kya hai", "fd maturity se pehle todni hai",
                     "senior citizen ko fd pe kitna interest", "meri fd kab mature hogi",
                     "ek saal ke liye {amt} ki fd karni hai", "fd jaldi band karne pe penalty lagegi kya"],
    },
    "net_banking_locked": {
        "en": ["My net banking is locked", "Forgot net banking password", "Internet banking user ID blocked",
               "Cannot log in to {bank} net banking", "Reset login password for internet banking", "Net banking OTP not coming",
               "Too many wrong attempts, account locked online",
               "I can't sign in to the bank website anymore", "Locked out of internet banking after password errors"],
        "hi": ["मेरी नेट बैंकिंग लॉक हो गई", "नेट बैंकिंग का पासवर्ड भूल गया", "{bank} नेट बैंकिंग में लॉगिन नहीं हो रहा",
               "इंटरनेट बैंकिंग का पासवर्ड रीसेट करना है", "नेट बैंकिंग का ओटीपी नहीं आ रहा",
               "बैंक की वेबसाइट पर अब साइन इन नहीं हो रहा", "गलत पासवर्ड से इंटरनेट बैंकिंग बंद हो गई"],
        "hinglish": ["meri net banking lock ho gayi", "net banking ka password bhool gaya", "{bank} net banking me login nahi ho raha",
                     "internet banking password reset karna hai", "net banking ka otp nahi aa raha",
                     "bank website pe ab sign in nahi ho raha", "galat password se internet banking band ho gayi"],
    },
    "mobile_number_update": {
        "en": ["Change my registered mobile number", "Update phone number in bank account", "I am not getting SMS alerts",
               "Link new mobile number to my {bank} account", "My old number is lost, how to update?",
               "OTP going to old number",
               "Register my mobile for banking alerts", "I changed my SIM, bank still has the old number"],
        "hi": ["रजिस्टर्ड मोबाइल नंबर बदलना है", "बैंक खाते में फोन नंबर अपडेट करना है", "एसएमएस अलर्ट नहीं आ रहे",
               "{bank} खाते से नया मोबाइल नंबर जोड़ना है", "ओटीपी पुराने नंबर पर जा रहा है",
               "बैंकिंग अलर्ट के लिए मोबाइल रजिस्टर करना है", "सिम बदल दी पर बैंक में पुराना नंबर है"],
        "hinglish": ["registered mobile number change karna hai", "bank account me phone number update karna hai",
                     "sms alerts nahi aa rahe", "{bank} account se naya mobile number link karna hai", "otp purane number pe ja raha",
                     "banking alerts ke liye mobile register karna hai", "sim badal di par bank me purana number hai"],
    },
    "complaint_escalation": {
        "en": ["I want to file a complaint", "My issue is not resolved for {days} days", "Escalate my complaint to the manager",
               "How do I complain to the banking ombudsman?", "Nobody is responding to my complaint", "Give me a complaint reference number",
               "Very bad service, I want to complain",
               "It has been {days} days and still no solution, escalate this", "Where can I raise a formal grievance?"],
        "hi": ["मुझे शिकायत दर्ज करनी है", "{days} दिन से मेरी समस्या हल नहीं हुई", "मेरी शिकायत मैनेजर तक पहुंचाइए",
               "बैंकिंग लोकपाल से शिकायत कैसे करें", "शिकायत का रेफरेंस नंबर दीजिए",
               "{days} दिन हो गए अभी तक हल नहीं, इसे आगे बढ़ाइए", "औपचारिक शिकायत कहां दर्ज करूं"],
        "hinglish": ["mujhe complaint file karni hai", "{days} din se meri problem solve nahi hui", "meri complaint manager tak bhejo",
                     "banking ombudsman me complaint kaise kare", "complaint ka reference number do",
                     "{days} din ho gaye abhi tak solution nahi, escalate karo", "formal grievance kaha raise karu"],
    },
    "account_opening": {
        "en": ["I want to open a savings account", "Open a zero balance account", "Documents needed to open an account",
               "Can I open an account online?", "Open a salary account with {bank}", "Minimum balance for savings account",
               "Open account for my child",
               "How do I start a new savings account digitally?", "What is needed for a new account, Aadhaar and PAN?"],
        "hi": ["मुझे बचत खाता खोलना है", "जीरो बैलेंस खाता खुलवाना है", "खाता खोलने के लिए कौन से दस्तावेज चाहिए",
               "क्या ऑनलाइन खाता खुल सकता है", "बचत खाते में न्यूनतम बैलेंस कितना",
               "डिजिटल तरीके से नया बचत खाता कैसे खोलें", "नए खाते के लिए आधार और पैन चाहिए क्या"],
        "hinglish": ["mujhe savings account kholna hai", "zero balance account khulwana hai", "account kholne ke liye kya documents chahiye",
                     "kya online account khul sakta hai", "savings account me minimum balance kitna",
                     "digital tarike se naya account kaise khole", "naye account ke liye aadhaar aur pan chahiye kya"],
    },
}


def fill(t: str, rng: random.Random) -> str:
    return t.format(app=rng.choice(APPS), bank=rng.choice(BANKS), amt=rng.choice(AMTS), days=rng.choice(DAYS))


def casual(s: str, rng: random.Random, lang: str) -> str:
    """Real typing: some messages lower-cased or without the final '?'."""
    if lang == "en" and rng.random() < 0.3:
        s = s.lower()
    if rng.random() < 0.3:
        s = s.rstrip("?")
    return s


def main():
    rng = random.Random(0)
    OUT.mkdir(parents=True, exist_ok=True)
    # Per intent and language: the last two templates are test, the one before is validation
    # (model selection and the confidence temperature), the rest train. No phrasing is shared.
    train, val, test = [], [], []
    for intent, by_lang in T.items():
        for lang, temps in by_lang.items():
            for i, t in enumerate(temps):
                split = test if i >= len(temps) - 2 else val if i == len(temps) - 3 else train
                n = 3 if split is not train else 8 if "{" in t else 2
                seen = set()
                for _ in range(n * 3):
                    s = casual(fill(t, rng), rng, lang)
                    if s in seen:
                        continue
                    seen.add(s)
                    split.append({"text": s, "label_text": intent, "lang": lang, "synthetic": True})
                    if len(seen) >= n:
                        break
    for rows in (train, val, test):
        rng.shuffle(rows)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                           encoding="utf-8")
    (OUT / "README.md").write_text(
        "# india_banking (SYNTHETIC)\n\nWritten for the AnuLM demo by `experiments/make_india_banking.py`: "
        f"{len(T)} intents of Indian retail banking (UPI, IFSC, KYC, Aadhaar, EMI, ...) in English, Hindi and "
        f"Hinglish. {len(train)} train / {len(val)} val / {len(test)} test messages. Not customer data. Val and test use "
        "phrasings that appear in no other split, so the score measures generalisation to new wordings of these "
        "intents, not accuracy on real traffic.\n", encoding="utf-8")
    from collections import Counter
    print(f"{len(T)} intents; train {len(train)}, val {len(val)}, test {len(test)}; test by lang {dict(Counter(r['lang'] for r in test))}")


if __name__ == "__main__":
    main()
