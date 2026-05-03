
import streamlit as st
import pandas as pd
import pickle
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

# =====================================================
# CUSTOM CLASSES (MUST MATCH TRAINING)
# =====================================================

class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['meal'] = X['meal'].str.strip()
        X['country'] = X['country'].str.upper()
        X['market_segment'] = X['market_segment'].str.strip()

        X['children'] = X['children'].fillna(0)
        X['country'] = X['country'].fillna('Unknown')
        X['agent'] = X['agent'].fillna(0)

        if 'company' in X.columns:
            X = X.drop('company', axis=1)
        if 'reservation_status' in X.columns:
            X = X.drop('reservation_status', axis=1)

        X['adr'] = X['adr'].clip(0, 5000)
        X['children'] = X['children'].astype(int)

        if 'reservation_status_date' in X.columns:
            X['reservation_status_date'] = pd.to_datetime(X['reservation_status_date'], errors='coerce')

            X['year'] = X['reservation_status_date'].dt.year
            X['month'] = X['reservation_status_date'].dt.month
            X['day'] = X['reservation_status_date'].dt.day

            X = X.drop('reservation_status_date', axis=1)

        X['total_people'] = X['adults'] + X['children'] + X['babies']
        X['total_nights'] = X['stays_in_weekend_nights'] + X['stays_in_week_nights']

        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, vt_threshold=0.01, corr_threshold=0.9, mi_threshold=0.01):
        self.vt_threshold = vt_threshold
        self.corr_threshold = corr_threshold
        self.mi_threshold = mi_threshold

    def fit(self, X, y):
        # convert to array safely
        X = np.array(X)

        # Variance threshold
        self.vt = VarianceThreshold(self.vt_threshold)
        X_vt = self.vt.fit_transform(X)

        # Correlation filter
        corr = np.corrcoef(X_vt, rowvar=False)
        upper = np.triu(np.ones(corr.shape), k=1).astype(bool)

        self.to_drop = set()
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[1]):
                if abs(corr[i, j]) > self.corr_threshold:
                    self.to_drop.add(j)

        keep_mask = np.array([i not in self.to_drop for i in range(X_vt.shape[1])])
        X_corr = X_vt[:, keep_mask]

        # Mutual information
        mi = mutual_info_classif(X_corr, y, random_state=42)

        self.mi_mask = mi > self.mi_threshold
        self.keep_mask = keep_mask

        return self

    def transform(self, X):
        X = np.array(X)

        X_vt = self.vt.transform(X)

        X_corr = X_vt[:, self.keep_mask]

        X_final = X_corr[:, self.mi_mask]

        return X_final

# =====================================================
# LOAD MODEL
# =====================================================
model = pickle.load(open("hotel_booking_pipeline.pkl", "rb"))

st.title("🏨 Hotel Cancellation Predictor")

# =====================================================
# INPUTS
# =====================================================

hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
lead_time = st.number_input("Lead Time", 0, 500, 50)

adults = st.number_input("Adults", 0, 10, 2)
children = st.number_input("Children", 0, 10, 0)
babies = st.number_input("Babies", 0, 10, 0)

weekend = st.number_input("Weekend Nights", 0, 10, 1)
week = st.number_input("Week Nights", 0, 30, 2)

meal = st.selectbox("Meal", ["BB","HB","FB","SC","Undefined"])
country = st.text_input("Country", "PRT")

market_segment = st.selectbox("Market Segment",
    ["Direct","Corporate","Online TA","Offline TA/TO","Groups","Aviation"])

distribution_channel = st.selectbox("Distribution Channel",
    ["Direct","Corporate","TA/TO","GDS"])

deposit_type = st.selectbox("Deposit Type",
    ["No Deposit","Refundable","Non Refund"])

customer_type = st.selectbox("Customer Type",
    ["Transient","Contract","Transient-Party","Group"])

adr = st.number_input("ADR", 0.0, 500.0, 100.0)

booking_changes = st.number_input("Booking Changes", 0, 10, 0)
special_requests = st.number_input("Special Requests", 0, 5, 0)

# =====================================================
# BUILD INPUT (FIXED FULL SCHEMA)
# =====================================================

def build_input():

    df = pd.DataFrame([{

        "hotel": hotel,
        "lead_time": lead_time,

        "arrival_date_year": 2017,
        "arrival_date_month": "July",
        "arrival_date_week_number": 27,
        "arrival_date_day_of_month": 1,

        "stays_in_weekend_nights": weekend,
        "stays_in_week_nights": week,

        "adults": adults,
        "children": children,
        "babies": babies,

        "meal": meal,
        "country": country,

        "market_segment": market_segment,
        "distribution_channel": distribution_channel,

        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,

        "reserved_room_type": "A",
        "assigned_room_type": "A",

        "booking_changes": booking_changes,
        "deposit_type": deposit_type,

        "agent": 0,
        "company": 0,

        "days_in_waiting_list": 0,

        "customer_type": customer_type,

        "adr": adr,

        "required_car_parking_spaces": 0,
        "total_of_special_requests": special_requests,

        "reservation_status": "Check-Out",
        "reservation_status_date": "2017-01-01"
    }])

    return df

# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict"):

    df = build_input()

    # IMPORTANT: must run cleaner manually
    df = model.named_steps["cleaner"].transform(df)

    pred = model.predict(df)[0]

    if pred == 1:
        st.error("❌ Cancelled")
    else:
        st.success("✅ Not Cancelled")
