# How to Win the IOS14 User Acquisition Battle? – Matej Lancaric: User Acquisition Expert to Grow Mobile Games Globally 🦄🕺

![ios14 User Acquisition Battle](https://lancaric.me/wp-content/uploads/2023/05/lancito_cyberpunk_ninja_slashing_numbers_in_the_post_apocalypti_381419f1-0d30-460a-b533-10fcd0a303dd.png) 

# How to Win the IOS14 User Acquisition Battle?

Matej Lančarič,
September 19 2022

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

Success lies in adapting strategies to overcome challenges introduced by privacy changes like the App Tracking Transparency (ATT) framework and IDFA loss. These changes limit data availability, making SKAdNetwork (SKAN) the cornerstone for attribution and campaign optimization in mobile games. By focusing on accurate conversion schemas—combining in-app events with revenue brackets—developers can ensure effective tracking and minimize null values in SKAN post-backs. Consolidating audience groups, such as lookalike audiences and interest-based segments, helps optimize bids and achieve the minimum install thresholds required for privacy compliance.Evaluating campaign performance requires patience due to SKAN reporting delays, but monitoring key metrics like Cost Per Install (CPI) and Return on Ad Spend (ROAS) ensures actionable insights. Platforms like Facebook and TikTok provide tailored recommendations, such as maintaining a daily install count above thresholds to clear privacy limits. Winning this battle demands combining creative formats, strategic audience grouping, and precise data analysis, ensuring mobile game campaigns remain competitive and effective despite the constraints of iOS14’s privacy landscape.

## How Has Apple’s SKAdNetwork Transformed Mobile UA and Campaign Measurement on iOS?

Apple’s SKAdNetwork Transformed Mobile UA and Campaign Measurement on iOS in following ways:

Following the announcement of ATT, Apple’s privacy-centric attribution framework SKAdNetwork (SKAN) is your new boss. Since the beginning, everybody said most iOS users will deny access to their user-level data in versions 14.5 and above so you are pretty much fucked. After some time, we found out that [40+% players actually share IDFA](https://youtu.be/2NE3ls6nVu4).

The ugly truth still is that you rely on SKAN to measure the success of your campaigns. **I do too!**

SKAN has turned [mobile UA](https://lancaric.me/) upside down on iOS, introducing completely new mechanisms to balance between data privacy and marketing measurement.

Expect delays in your reporting from 24 – 72 hours from the actual install or subsequent in-app conversion event. This is a result of Apple’s delay framework. Conversion value updates from your app may delay reporting beyond 48 hours depending on your conversion model. Currently, limit conversion values to events that occur within the 24 hours following the first app open. **This is the golden rule for iOS profitable campaigns**! This may mean not configuring some events that tend to occur days or weeks after an install

## How Can the Right SKAN Conversion Schema Boost iOS User Acquisition for Mobile Games?

Right SKAN Conversion Schema can Boost iOS User Acquisition for Mobile Games by following ways:

**Wrong conversion schema?**

This is always the first thing I am checking when starting to work on a new game. How does the conversion schema setup looks like. It can really make or break your whole iOS UA efforts.

I’ve been testing this out since the beginning of the SKAN. Oh boy, how much fun did I have. Going from funnel schema, to revenue schema (which actually worked!) to custom schema where I combine both funnel events with revenue brackets.

Why combination? Revenue brackets are important for you to be able to run Value optimised campaigns!

* which ones do people use? What is recommended?

![](https://lancaric.me/wp-content/uploads/2022/09/Screen-Shot-2022-09-04-at-21.15.32-1024x408.png)  
![](https://lancaric.me/wp-content/uploads/2022/09/Screen-Shot-2022-09-04-at-21.15.00-1024x440.png)  
![](https://lancaric.me/wp-content/uploads/2022/09/Screen-Shot-2022-09-04-at-21.14.31-1024x443.png)  
![](https://lancaric.me/wp-content/uploads/2022/09/Screen-Shot-2022-09-04-at-21.14.00-1024x437.png)  
![](https://lancaric.me/wp-content/uploads/2022/09/Screen-Shot-2022-09-04-at-21.12.47-1024x430.png)

* **For more genres and benchmarks look here**: <https://www.appsflyer.com/benchmarks/gaming-action/skad/global/2022-05-01/>

As you can see on the screenshots from different genres, in-app-events + revenue (brackets) is the most used conversion schema out there. **Test it on your end as well, don’t use it blindly.**

After being 300% sure the SKAN conversion schema is correct, I always look at the [campaigns performance](https://lancaric.me/12-tips-for-killer-user-acquisition-operations/). Usually campaigns are not passing the privacy threshold therefore **getting 0 conversions.**

## What Is the SKAdNetwork Privacy Threshold and How Does It Impact Conversion Attribution?

SKAdNetwork Privacy Threshold is play a key role in conversion attribution. When a user installs and launches an app on iOS after viewing an ad, Apple requires a certain number of conversions to occur before providing attribution for installs with conversion values for an ad to a given Ad Network. This is called the SKAdNetwork Privacy Threshold, which helps prevent advertisers from identifying a unique user.

Apple has said the post-back conversion may include a conversion value and the source app’s ID if Apple determines that providing the values meets their privacy threshold.

When Apple believes there’s a risk of identifying a user the conversions are deemed “NULL” conversions in the SKAdNetwork post-back.

## Why Do In-App Event Optimization Campaigns Struggle with SKAN Postbacks and Privacy Thresholds?

For campaigns optimizing for in-app events: this impacts both optimization and reporting for in-app events as the Ad Networks gets limited data for the purpose of optimizing performance of an AEO campaign.

**You are running an iOS Purchase campaign on Facebook, but you can’t really see any postbacks. Right?**

![](https://lancaric.me/wp-content/uploads/2022/09/AppsFlyer-13-1024x475.png)  
![](https://lancaric.me/wp-content/uploads/2022/09/AppsFlyer-9-1024x176.png)

Let me tell you this -> 0 SKAN postbacks on FB = not passing the privacy threshold! But this is not an issue only on Facebook.

## What Are the Privacy Threshold Requirements for SKAdNetwork Campaigns Across Different Ad Networks?

Privacy Threshold Requirements for SKAdNetwork Campaigns Across Different Ad Networks are as followed:

Each network has its own internal logic when it comes to figuring out the privacy threshold.

To avoid null values for SKAd post-backs, you need to ensure that you get a minimum number of installs above Apple’s privacy threshold.

They offer their advertisers guidance accordingly; each with its own set of campaigns hierarchy, ad-sets, publishers etc.

**Snapchat** recommends driving at least 75+ Installs per Ad Set per Day for optimal measurement. They have observed a positive impact on clearing privacy thresholds and reducing Withheld conversion values when the number of daily attributed installs is high.

**The Unity** team recommends aiming for a minimum of 30 installs per day. If you see a lot of null values, we recommend reducing the number of campaigns and/or creatives per campaign. While you have a limit of 50 live campaigns and/or creative packs per app, using all of these may cause you to be below the minimum of 30 installs per day for the privacy threshold.

**Facebook**, for example, aims for at least **88 app installs per day**: If you don’t have at least 88 app installs per campaign per day, you’ll likely see null conversions in your reporting. Combining ad sets and campaigns can help meet the 88 app install threshold to minimize null conversions. You can also consider increasing budgets and/or moving up the funnel to more populous or more frequent events.

**TikTok** recommends a daily target of 90+ installs per iOS 14 campaign. If you are seeing much fewer installs for your iOS 14 campaigns, they recommend reviewing your campaign specs.

**Google** campaigns are using modeled conversions which means that modeled conversions may take up to 5 days to appear, and this is expected behavior. This modeling delay is in addition to existing conversion delays. Also be aware that reporting for recent campaign performance will continue to be affected by modeling lags, and you should continue to be prepared for performance fluctuations.

In the world of User acquisition, 72 hours is already too long (this is the time you are not able to evaluate anything on other channels) not talking about 5 days delay. **You need to get at least 100 installs per day per campaign.**

## What Could you do to Start Getting Postbacks and Meet the Privacy Threshold?

To start getting postbacks and meet the privacy threshold you can do following things:

### Increase budget

Increase your budget to run campaigns with considerably high budgets. Look at your CPIs on Android, multiply them 5-8x and calculate the budget. **Simplified, if you run $150-200/day on Android, on iOS you need to start at least on $500-600/day.**

### Consolidate Countries

**Consolidate countries and create geo Tier buckets. My current Tiers example:**

**T1**: US, UK, CA, DE, FR, AU, KR, JP, CH

**T2:** DK, FI, NO, SE, NL, ES, IT, HK, SG, NZ

**T3:** PL, CZ, SK, BE, AT, IE, TR, UAE, SA, TW, TH

**ROW**

I usually group all tier 1 countries together not looking at the language, but only LTV (lifetime value), then create tier 2 segment and Rest of the World. This is based on LTV for a specific game.

**Don’t copy, do your own homework!**

### Use broader Targeting Strategies

Use broad targeting as much as possible, and consolidate lookalikes, interest or behavior with high overlap into larger groups.

### Facebook AAA Strategy

With Facebook consider switching campaigns to AAA (Automated app ads) structure to help overcome the privacy threshold and start getting postbacks.

Use AAA campaign with MAI+Purchase optimization. As such, 50% of the time the campaign will optimise for installs and other 50% will optimise for purchase. Using this campaign type especially on iOS to get lower CPIs, but also quality players.

### iOS Campaign Optimization

You have a limited number of [UA campaigns](https://lancaric.me/survivor-io-global-launch-ua-case-study/) so consider grouping countries, consolidating lookalike and interest group audiences where you have similar bids or similar ARPU.

Don’t make campaign adjustments in the first 72 hours after launching the iOS 14 campaign.

Wait until you see first data, then when evaluating overall campaign performance, please wait for an additional 24 hours (ideally 72 hours) so that the campaign can fully receive all  SKAN conversion post-backs.

**Expect a longer learning phase for the iOS14 campaigns due to SKAN delays and IDFA loss. Super important!**

When you do all these things this is how it should look!

![](https://lancaric.me/wp-content/uploads/2022/09/UA-channel-comparison-final-1-1024x332.png)  
![](https://lancaric.me/wp-content/uploads/2022/09/Skan-Dashboard-final-1-1024x660.png)  
![](https://lancaric.me/wp-content/uploads/2022/09/AppsFlyer-11-1024x423.png)  
![](https://lancaric.me/wp-content/uploads/2022/09/AppsFlyer-12-1024x254.png)

All good, right?

## Anything that I missed?

Facebook  
Twitter  
LinkedIn  
 [Subscribe now](https://www.youtube.com/channel/UCpu7cyygnCCCkUEMG2-JiHQ)

## **Subscribe to Brutally Honest Newsletter**



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
