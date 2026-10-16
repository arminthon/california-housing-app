import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ۱. بارگذاری مدلی که قبلاً ذخیره کردیم (housing_model.pkl)
# از cache@ استفاده می‌کنیم تا مدل فقط یک بار لود بشه و برنامه سریع‌تر کار کنه
@st.cache_resource
def load_model():
    return joblib.load("housing_model.pkl")


model = load_model()

# ۲. عنوان و توضیحات صفحه وب
st.title("🏡 پیش‌بینی قیمت مسکن در کالیفرنیا")
st.write(
    "این اپلیکیشن با استفاده از مدل **Random Forest** و کتابخانه **Streamlit** قیمت خانه را بر اساس ویژگی‌های منطقه پیش‌بینی می‌کند."
)

st.divider()

# ۳. طراحی بخش ورودی‌ها (اسلایدرها برای کاربر)
st.header("مقادیر ورودی منطقه را وارد کنید:")

# اسلایدر برای میانگین درآمد منطقه
med_inc = st.slider(
    "میانگین درآمد منطقه (MedInc)",
    min_value=0.5,
    max_value=15.0,
    value=3.5,
    step=0.1,
)

# اسلایدر برای سن خانه
house_age = st.slider(
    "میانگین سن خانه‌ها در منطقه (HouseAge)",
    min_value=1.0,
    max_value=50.0,
    value=20.0,
    step=1.0,
)

# اسلایدر برای تعداد اتاق‌ها
ave_rooms = st.slider(
    "میانگین تعداد اتاق‌ها (AveRooms)",
    min_value=1.0,
    max_value=10.0,
    value=5.0,
    step=0.1,
)

st.divider()

# ۴. ساخت دکمه پیش‌بینی و پردازش ورودی‌ها
if st.button("پیش‌بینی قیمت خانه 🚀"):
    # آماده‌سازی داده‌های ورودی به فرمتی که مدل می‌فهمد (به صورت DataFrame با نام ستون‌ها)
    input_data = pd.DataFrame(
        {
            "MedInc": [med_inc],
            "HouseAge": [house_age],
            "AveRooms": [ave_rooms],
        }
    )

    # پیش‌بینی قیمت با استفاده از مدلی که لود کردیم
    prediction = model.predict(input_data)

    # چون واحد دیتاست صدهزار دلار است، قیمت را در ۱۰۰,۰۰۰ ضرب می‌کنیم تا عدد واقعی‌تر نمایش داده شود
    predicted_price = prediction[0] * 100000

    # نمایش نتیجه به کاربر با استایل زیبا
    st.success(
        f"💰 قیمت تخمینی این خانه حدوداً: **${predicted_price:,.0f} دلار** است."
    )
# 5. Real chart: effect of income on predicted house price
st.subheader("💰 Effect of Income on Predicted House Price")
st.write("Each point shows the model's predicted house price for a given median income, keeping other factors fixed.")

# we create 20 income values from 1 to 10 (each unit = $10,000/year)
incomes = np.linspace(1, 10, 20)   # 20 equally spaced numbers between 1 and 10

# build the input table: MedInc changes, HouseAge=10 and AveRooms=5 stay fixed
input_data = pd.DataFrame({
    'MedInc': incomes,      # the changing variable
    'HouseAge': [10] * 20,  # fixed value so results are comparable
    'AveRooms': [5] * 20    # fixed value so results are comparable
})

# ask the trained model to predict the price for each income
predictions = model.predict(input_data)

# put income and price in one table for the chart
chart_df = pd.DataFrame({
    'Median Income ($10k/year)': incomes,
    'Predicted Price ($100k)': predictions
}).set_index('Median Income ($10k/year)')

# draw the chart — x axis = income, y axis = price
st.line_chart(chart_df)
