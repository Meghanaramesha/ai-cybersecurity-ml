# 8 Must-haves before the post-IDFA Soft Launch – Matej Lancaric: User Acquisition Expert to Grow Mobile Games Globally 🦄🕺

![Post-IDFA Soft Launch Must Have](https://lancaric.me/wp-content/uploads/2023/05/must-have-rubber-stamp-vector-12434152.png) 

# 8 Must-haves before the post-IDFA Soft Launch

Matej Lančarič,
April 19 2023

♻️ Sharing is caring! 🫶

LinkedIn
Twitter 
Reddit 
Facebook  
WhatsApp
Telegram



[![](https://lancaric.me/wp-content/uploads/2025/02/4-150x150.png)](https://lancaric.me/author/lanco/)

##### [Matej Lančarič](https://lancaric.me/author/lanco/)

UA Mobile Growth Extraordinaire ·

Helping you stay 2.5 steps ahead of the games industry. Don't be too serious, except about UA.  
[Subscribe to my Brutally Honest newsletter!](https://lancaric.substack.com/)

8 Must-haves before the post-IDFA Soft Launch are essential for ensuring the success of user acquisition campaigns in the evolving mobile games ecosystem. With the IDFA deprecation and the introduction of App Tracking Transparency (ATT), accurate tracking now relies heavily on SKAdNetwork (SKAN) integration and a well-defined conversion schema. To meet investor expectations, mobile game developers must set up analytics tools to capture key metrics like installs, Cost Per Install (CPI), and Return on Ad Spend (ROAS). Proper creative testing is critical to ensure ad formats meet platform requirements on networks like Facebook, Google, and Unity Ads.

Preparing for a soft launch also involves consolidating user acquisition (UA) strategies, such as organizing audience groups and optimizing bids to meet privacy thresholds. Ensuring compliance with SKAN reporting by configuring in-app event tracking and revenue brackets minimizes null values in post-backs. Additionally, aligning campaign performance data with investor-ready reports provides credibility during this critical phase. These eight steps ensure that mobile game developers can navigate post-IDFA challenges and maximize the impact of their UA efforts during the soft launch.

BUT!

There are so many things you actually need to do before you can launch a [UA campaign](https://lancaric.me/11-tips-for-killer-user-acquisition-ops-q1-2023/)

Everybody talks about how to launch campaigns on specific networks. There is a ton of material on the web on how to launch campaigns on Facebook, Google Ads, Unity, etc. Yet again, no one talks about what needs to be done before you successfully launch a first campaign.

You must complete a few key steps before running ads for mobile app installs and tracking app installs, which takes some time. This should be kept in mind, and today we will discuss the “must haves” that must be done before we start the first campaign.

## What Are the Steps and Challenges of Publishing an App Successfully on App Stores?

Steps and Challenges of Publishing an App Successfully on App Stores are as followed:

![](https://lancaric.me/wp-content/uploads/2022/07/toppng.com-roll-safe-roll-safe-1241x1037-1-1024x856.png)

## **1 – Publish your game to the Store**

For a successful soft launch, publishing the app on the Store is crucial. Really? **Badum TSS**! Jokes aside! The app is ready, and the developer uploaded the application file to a store. Remember that the approval process takes one to two business days on average.

*“After submitting an iOS app for review, you can view its status in My Apps in iTunes Connect or the iTunes Connect app for iPhone and iPad. You can view the status in the Google Play Console for an Android app submission. For Google Play, an app review can take a few hours up to 7 days. If your application is incomplete, Google or Apple may take more time to process it, or your application may be rejected.*

*For the Apple App Store, 50% of apps are reviewed in 24 hours, and over 90% are reviewed in 48 hours. Once your application has been reviewed, its status will be updated, and you will receive a notification.”*

Let’s say the app is published, so you successfully upload an APK file select a language, and enter a name, short and long description, hi/res icon, feature graphics, etc., for your app. Implementing an [MMP SDK is crucial as a next step](https://lancaric.me/how-to-choose-an-mmp-in-a-privacy-centric-mobile-ecosystem/) for me. You can do it before submitting the app to the Store. If you do it after publishing the app, you need to release it with a new build, and the app will be in review for a couple of hours or days, depending on the platform.

There is also **extra fun** when you are trying to publish your first game/app on your developer account. First games always take longer, **sometimes even 7 days**. Be prepared for it!

![](https://lancaric.me/wp-content/uploads/2023/03/how-to-publish-app-on-store-2.png)

## Why Should You Implement an MMP from the Start of Your User Acquisition Journey?

You Should Implement an MMP from the Start of Your User Acquisition Journey for the following reason:

Some people will convince you that implementing MMP is a good idea once you start scaling a game. In my humble opinion, its always better to have the right and accurate data from the beginning of the UA journey. It’s time-saving and, in the long run, cost-saving. MMP is an economical solution; a universal SDK connects advertisers to the entire mobile ecosystem of many ad networks without relying on costly, time-intensive development resources. More about how to choose and why it’s good to have MMP you can be read [here](https://lancaric.me/how-to-choose-an-mmp-in-a-privacy-centric-mobile-ecosystem/).

You need to integrate MMP SDK into your app and define your app on Meta Ads, TikTok, Google, etc. With MMP, you have one SDK for everything, so for instance, you don’t need to integrate your app with Facebook’s SDK for mobile attribution to launch a successful campaign.

![](https://lancaric.me/wp-content/uploads/2023/01/ihave.jpeg)

## **No bullshit gaming podcast two & a half gamers break!** two & half gamers gaming podcast

## What Are the Integration Setup Requirements for Different Ad Networks?

Integration Setup Requirements for Different Ad Networks are as followed:

Each Ad network has a different setup where you must define your integration settings before starting advertising. In my opinion, the fastest process is on Google ads or Facebook. You can find the instructions [here](https://lancaric.me/how-to-ua-create-app-on-facebook-meta/). Just note that this also takes some additional time.

![](https://lancaric.me/wp-content/uploads/2023/04/stiahnut.png)

## How to Properly Set Up Post backs to Avoid Duplicate Payments and Installs?

To Properly Set Up Post backs to Avoid Duplicate Payments and Installs, you need to do following things:

All the MMPs have great notes on setting up post back. Just follow the instruction from their guidelines. Pay attention how are you sending post backs. Should be only from your MMP and not from other SDKs you implemented. Especially for Facebook, as mentioned above, you didn’t implement both FB and MMP SDK.

If you implemented both SDKs, verify that the MMP SDK is initialized at app launch, before the Facebook Ads SDK, and don’t configure events in the Facebook SDK or disable the Facebook in-app events mapping from your MMP.

You can avoid duplicating payments or installs.

As you can see in the screenshot below, we launched campaigns on one project, and after a couple of days, we noticed that the payments were **double counted** on Facebook (because of MMP + FB post back setup), so the ROAS was not as good as we thought.

![](https://lancaric.me/wp-content/uploads/2023/03/Screenshot-2023-03-17-at-09.35.44.png)

## How Can SKAN 4.0 Conversion Mapping Optimize Mobile Game Campaigns?

SKAN 4.0 Conversion Mapping Optimize Mobile Game Campaigns in following ways:

Most MMPs offer conversion mapping for SKAN campaigns. Now we have a [SKAN 4.0](https://youtu.be/edfxCUnVOnI) that offers three postbacks spread over time (up to 35 days). Overall, the multiple postback system gives advertisers a broader perspective on their users’ engagement, enabling them to craft better optimization and reporting plans. Since each variable (event, revenue, activity timer, and funnel) runs against mapping capacity, you must find the right balance. Depending on the needs of an App, conversion mapping can focus on in-app events with a giant post-install activity timer, events funnel, revenue, or a combination of variables. [Why combination?](https://lancaric.me/user-acquisition-ios14-privacy-threshold-pass/) Revenue brackets are important for running Value optimized campaigns!

**Note:**

When a user installs and launches an app on iOS after viewing an ad, Apple requires a certain number of conversions to occur before providing attribution for installs with conversion values for an ad to a given Ad Network. This is called the SKAdNetwork Privacy Threshold, which helps prevent advertisers from identifying a unique user. To avoid null values for SKAd post-backs, you need to ensure that you get a minimum number of installs above Apple’s privacy threshold.

Facebook – min. installs per day per campaign 88

Unity – min. installs per day per campaign 30

TikTok – min. installs per day per campaign 90

Google – min. installs per day per campaign 100

Until SKAN 4.0 hits us in the face. Then its going to be called **the****Crowd anonymity.**

![](https://lancaric.me/wp-content/uploads/2023/04/1-Wait-what-meme.png)

## How Can Proper Payment Setup Prevent Campaign Disruptions in User Acquisition?

Proper Payment Setup Prevents Campaign Disruptions in User Acquisition by following ways:

Don’t forget about the payment setup on your **UA channels**! As you start scaling campaigns, adding a secondary payment method is necessary. This prevents the Ad network from stopping your campaigns if your primary card expires, reaches the monthly limit, or is blocked for any other reason. If you are an agency and going to spend more money, you can avail monthly credit line with most of the Ad networks. Be careful and watch out for spending, especially when you start scaling. I was often in a situation when we started scaling campaigns, and after two weeks, we hit the credit line, and campaigns stopped spending. That means restarting the learning phase on campaigns, so you de facto threw money into the trash.

![](https://lancaric.me/wp-content/uploads/2022/09/moneyu.gif)

**Note**

If you choose Facebook as your go-to advertising platform, the billing threshold amount varies based on your billing history. So if you have a new account on Meta without a payment history, it’s better to call for a credit line from a Facebook rep. When you start, this threshold will be pretty low (usually $25), and you’ll be billed every time you spend $25 on Facebook Ads. Your threshold will automatically increase as you keep spending and your payments are correctly processed.

**Bonus tip: Create multiple ad accounts,** so if you main ad account gets banned, you can run your campaigns on others**.** Why would you get a ban? I have no idea, same goes with Facebook. They have no idea why, but they will ban you from advertising because they can. And they will.

## How to Create Winning Ad Creatives for Post-IDFA Campaigns?

To create winning ad creatives for post IDFA campaigns you need to follow following things:

Prepare creatives for your first campaign. Check competitors, and do your homework. Start with the basics of creatives. Snatching the viewer’s attention in the first three seconds is essential. Once they’re invested, you want to showcase the gameplay and make sure that it connects with the game’s storyline and communicates an understanding of the player, as players are looking for relevant content. [Full guide for creative framework can be found here!](https://lancaric.me/creative-framework-in-the-post-idfa-world/)

Social advertising algorithms are rapidly evolving as we move from deterministic to probabilistic modeling. In post-IDFA iOS 14.5, the creative must **pull you in** much more. Write a copy that is both easy to read and enticing. You’ll need a strong CTR and a funnel that maximizes conversions to come out on top in the ad auction. Let’s optimize the system by allowing only high-intent customers to click on the ads.

## How Can a Tech Test Campaign Save Costs in User Acquisition?

Tech Test Campaign Save Costs in User Acquisition by following ways:

Let’s start with your first campaign. I always start with a **tech test** campaign targeted at the Philippines or another cheap country regarding cost per install. Because no matter how good your setup may be, you should always check it before you start spending real money!

How to make a data check test and potentially save millions? You can read more [here](https://lancaric.me/how-to-check-your-games-ua-data/).

![](https://lancaric.me/wp-content/uploads/2021/12/Untitled-presentation-61.png)

## What Are the Essential Steps to Prepare for a Successful Mobile Game Ad Campaign Launch?

Essential Steps to Prepare for a Successful Mobile Game Ad Campaign Launch are as followed:

1. **The App needs to be published in stores**Be sure the App is published, not just submitted for review.
2. **MMP already implemented in your App**MMP SDK is crucial for campaign evaluation, saving time and additional costs. Remember that without an MMP, you rely exclusively on your ad platform for data.
3. **App defined in the specific Ad network**Don’t forget that this is time-consuming, and it’s good to have everything prepared before launching a campaign.
4. **Right postback setup**Tracking postbacks is crucial for advertisers and media sources, as they provide the necessary data to make informed decisions when optimizing ad campaigns and conversion rates. Underestimating their importance can lead to missed opportunities for data-driven insights.
5. **Right SKAN setup**Choose the conversion model that best matches your KPIs and campaign optimization processes.
6. **Don’t forget about the payment/credit line setup.**Be sure you have a backup payment method to prevent the Ad network from stopping your campaigns when some unexpected issue occurs with your primary card.
7. **Creatives & Ad copy prepared**You can use screenshots or videos from the store for the first campaign. 1-2 Ad copy inspired by your long description.
8. **Let’s launch a data test campaign.**Make sure you have everything set up right. It’s better to spend a few bucks initially than to find out later that your data is wrong.

![](https://lancaric.me/wp-content/uploads/2022/04/claire-dancing.gif)

## **Anything I missed?**

Oh wow! You made it until here! You must be very engaged. I like that type of players.. Ehm, people!

Please share this article with your industry friends. It would mean a world to me.

Also, subscribe to my newsletter. **It’s so honest it might actually annoy you**. If you are easily annoyed, please don’t subscribe.

Facebook  
Twitter  
LinkedIn  
 [Subscribe now](https://www.youtube.com/channel/UCpu7cyygnCCCkUEMG2-JiHQ)

## **Subscribe to Brutally Honest Newsletter**

UA Insights so honest it might actually annoy you.



[![](https://lancaric.me/wp-content/uploads/2025/02/4-150x150.png)](https://lancaric.me/author/lanco/)

##### [Matej Lančarič](https://lancaric.me/author/lanco/)

UA Mobile Growth Extraordinaire ·

Helping you stay 2.5 steps ahead of the games industry. Don't be too serious, except about UA.  
[Subscribe to my Brutally Honest newsletter!](https://lancaric.substack.com/)

♻️ Sharing is caring! 🫶

LinkedIn
Twitter 
Reddit 
Facebook  
WhatsApp
Telegram
