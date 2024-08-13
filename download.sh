cd ../data_process/amazon/data

AMAZON_CATEGORIES=(\
    Amazon_Fashion \
    All_Beauty \
    Appliances \
    Arts_Crafts_and_Sewing \
    Automotive \
    Books \
    CDs_and_Vinyl \
    Cell_Phones_and_Accessories \
    Clothing_Shoes_and_Jewelry \
    Digital_Music \
    Electronics \
    Gift_Cards \
    Grocery_and_Gourmet_Food \
    Home_and_Kitchen \
    Industrial_and_Scientific \
    Kindle_Store \
    Luxury_Beauty \
    Magazine_Subscriptions \
    Movies_and_TV \
    Musical_Instruments \
    Office_Products \
    Patio_Lawn_and_Garden \
    Pet_Supplies \
    Prime_Pantry \
    Software \
    Sports_and_Outdoors \
    Tools_and_Home_Improvement \
    Toys_and_Games \
    Video_Games
)

for AMAZON_CATEGORY in "${AMAZON_CATEGORIES[@]}"; do
    echo $AMAZON_CATEGORY
    wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_${AMAZON_CATEGORY}.json.gz
done
