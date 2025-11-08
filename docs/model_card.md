# Model Card: Real-Time Lead Scoring System

## Model Details

**Model Type:** Random Forest Classifier  
**Version:** 1.0  
**Date:** November 2025  
**Developer:** Jordan Sinclair  
**Framework:** PySpark MLlib 3.5.0  
**License:** MIT

### Model Description

A machine learning pipeline that predicts the likelihood of a prospective homebuyer contacting a real estate agent based on their website browsing behavior and engagement patterns.

## Intended Use

### Primary Use Cases
- **Real-time lead prioritization** for real estate agents
- **Automated lead routing** to match high-intent buyers with available agents
- **Sales team efficiency** by focusing outreach on qualified leads

### Intended Users
- Real estate sales teams and agents
- Customer relationship management (CRM) systems
- Marketing automation platforms

### Out-of-Scope Uses
- Should not be used as the sole decision-maker for agent assignments
- Not intended for credit decisions or financial risk assessment
- Not designed for predicting home purchase decisions (only agent contact likelihood)

## Training Data

### Dataset
- **Source:** Kaggle Lead Scoring Dataset (adapted for real estate domain)
    - **Link**: https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset/
- **Size:** 9,240 records
- **Split:** 70% training, 30% held out for streaming simulation
- **Validation:** 80/20 split of training data for evaluation

### Features
**Behavioral Features:**
- `TotalVisits` - Number of property listings viewed
- `TotalBrowsingTime` - Time spent on website (minutes)
- `AvgListingsViewedPerSession` - Average properties per session

**Engagement Features:**
- `LeadCaptureChannel` - How the lead was captured (form, import, etc.)
- `ReferralSource` - Traffic source (Facebook, referral, etc.)
- `LastAction` - Most recent user action
- `FinalEngagementAction` - Last engagement type

**Demographic Features:**
- `City` - User location
- `Country` - User country
- `LeadStatusTag` - Lead status classification

### Target Variable
- `ContactedAgent` - Binary (0: No contact, 1: Contacted agent)

### Preprocessing
1. **Missing value imputation** using mean imputation for numerical features
2. **Bucketing** for `TotalVisits` (0-10, 10-20, ..., 300+)
3. **One-hot encoding** for all categorical features
4. **Feature vector assembly** combining all engineered features

## Model Performance

### Evaluation Metrics

| Metric    | Value  | Interpretation |
|-----------|--------|----------------|
| AUC-ROC   | 0.96   | Excellent discrimination between classes |
| Accuracy  | 0.86   | Correctly classifies 86% of leads |
| Precision | 0.88   | 88% of predicted high-priority leads are truly high-priority |
| Recall    | 0.86   | Captures 86% of actual high-priority leads |

### Performance Across Segments

The model performs consistently across:
- Various lead capture channels
- Multiple referral sources
- Different cities and countries

**Note:** Performance may degrade for:
- New cities not in training data
- Unusual browsing patterns (e.g., extremely high visit counts)
- New lead capture channels introduced after training

## Model Architecture

**Pipeline Stages:**
1. Imputer (mean imputation for all numeric features)
2. Bucketizer (TotalVisits → buckets)
3. String Indexers (7 categorical features)
4. One-Hot Encoders (7 categorical features)
5. Vector Assembler (combine all features)
6. Random Forest Classifier

**Hyperparameters:**
- Number of trees: 100
- Max depth: 10
- Min instances per node: 1 (default)
- Feature subset strategy: auto

## Limitations

### Known Limitations

1. **Temporal drift**: Model trained on historical data may not capture seasonal trends in home buying
2. **Cold start problem**: New users with minimal browsing history get less accurate scores
3. **Geographic bias**: Model may be biased toward cities with more training examples (Mumbai, Delhi)
4. **Feature limitations**: Doesn't capture external factors (interest rates, housing market conditions)
5. **Web Behavior Differences**: The dataset was adapted from interacions on an education websites, some differences may be noticed when streaming interactions from Zillow. 

### Technical Limitations

- **Latency**: Streaming pipeline processes batches (not individual events)
- **Scalability**: Tested on ~3k records; performance at millions of events/day unknown
- **Missing data**: Heavy reliance on imputation may reduce accuracy for sparse user profiles

### Fairness Considerations

- Model uses `City` and `Country` features, which could inadvertently create geographic bias
- No explicit fairness constraints applied during training
- Recommend monitoring for disparate impact across geographic regions

## Ethical Considerations

### Privacy
- Model uses aggregated behavioral data, not personally identifiable information (PII)
- User IDs are pseudonymized
- Complies with privacy-by-design principles

### Bias & Fairness
- **Risk**: Model may prioritize leads from certain cities if those leads had higher conversion rates historically
- **Mitigation**: Monitor lead score distributions across geographic regions
- **Recommendation**: Periodic audits to ensure equitable lead distribution

### Transparency
- Feature importance available from Random Forest model
- Predictions include probability scores (not just binary decisions)
- Real estate agents should review model recommendations, not blindly follow them

## Monitoring & Maintenance

### Recommended Monitoring

1. **Data Drift Detection**
   - Monitor feature distributions (especially TotalVisits, browsing time)
   - Alert if distributions shift significantly from training data

2. **Performance Monitoring**
   - Track conversion rates of high-scored leads
   - Compare predicted vs actual contact rates weekly

3. **Fairness Monitoring**
   - Monitor lead score distributions by city/region
   - Check for disparate impact across demographics

### Retraining Triggers

Retrain model when:
- AUC drops below 0.8 on validation data
- Significant data drift detected (KL divergence > threshold)
- New lead capture channels or cities introduced
- Business logic changes (e.g., new definition of "high-priority" lead)

**Recommended retraining frequency:** Monthly

## Deployment Considerations

### Production Requirements

- **Compute**: Spark cluster with 8+ cores recommended for real-time scoring
- **Latency**: Target <100ms per batch of leads
- **Storage**: Model artifacts ~50MB, requires S3/GCS for persistence
- **Dependencies**: PySpark 3.5+, Java 17

### A/B Testing

Before full deployment:
1. Shadow mode: Score leads but don't act on predictions
2. A/B test: 10% of leads scored by model, 90% by existing process
3. Gradual rollout: Increase to 50%, then 100% if performance validated

## Contact & Updates

**Maintained by:** Jordan Sinclair
**Repository:** https://github.com/smiley-maker/realtime-lead-scoring/tree/main
**Questions:** jordan.sinclair@du.edu

**Version History:**
- v1.0 (Nov 2025): Initial model with Random Forest classifier and all described features