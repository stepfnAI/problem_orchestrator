# AI-Powered Problem Orchestrator Pipeline

An intelligent machine learning pipeline designed for analyzing business data across multiple domains. This orchestrator helps organizations consolidate data from various sources, prepare it for machine learning, and train models to derive actionable insights.

🌟 Features

- **Multi-Source Data Processing**: Process and analyze data from multiple sources:
  - Financial/Billing data
  - Product usage metrics
  - Customer interactions
  - And more...

- **Intelligent Data Pipeline**:
  1. **Smart Data Gathering**
     - Multiple file format support (CSV, XLSX, JSON, Parquet)
     - AI-powered automatic file categorization
     - Initial data validation

  2. **AI-Powered Data Mapping**
     - Intelligent column mapping suggestions
     - Standard schema validation
     - Custom field mapping support
     - Problem-level granularity selection

  3. **Automated Data Cleaning**
     - Smart data type detection
     - Missing value handling strategies
     - AI-suggested cleaning rules
     - Interactive cleaning confirmation

  4. **Flexible Data Aggregation**
     - Multi-level aggregation options
     - AI-suggested aggregation methods
     - Customizable aggregation rules per column
     - Aggregation explanation support

  5. **Advanced Data Joining**
     - Two-phase joining process
     - Smart join key detection
     - Comprehensive join health validation
     - Join progress tracking

  6. **Data Splitting**
     - Automated train/validation/inference split
     - Time-based splitting support
     - Configurable splitting strategies

  7. **Model Training**
     - Multiple model support (XGBoost, Random Forest, LightGBM, CatBoost)
     - Automated feature selection
     - Performance metrics tracking
     - Interactive training progress

  8. **Model Selection**
     - AI-powered model recommendation
     - Performance comparison
     - Metric-based selection
     
  9. **Model Inference**
     - Batch inference support
     - Results visualization
     - Performance monitoring

- **Supported Use Cases**:
  - **Classification**: Binary classification problems
  - **Regression**: Numeric value prediction
  - **Forecasting**: Time-series prediction
  - **Clustering**: Unsupervised pattern discovery
  - **Recommendation**: Item suggestion systems

🚀 Getting Started

**Prerequisites**

- Python 3.9-3.11
- OpenAI API key

### Installation
1. Clone the repository:

```bash
git clone https://github.com/stepfnAI/problem_orchestrator.git
cd problem_orchestrator
```

2. Create and activate a virtual environment using virtualenv:

```bash
pip install virtualenv                # Install virtualenv if not already installed
virtualenv venv                       # Create virtual environment
source venv/bin/activate             # Linux/Mac
# OR
.\venv\Scripts\activate              # Windows
```

3. Install the package in editable mode:

```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY='your_openai_api_key'
```

🔄 Pipeline Workflow

1. **Start the Application**

```bash
# Windows
streamlit run .\orchestration\main_orchestration.py

# Linux/Mac
streamlit run orchestrator/app.py
```


2. **Follow the Step-by-Step Process**:
   - Upload your data files
   - Confirm automatic categorization
   - Review and adjust column mappings
   - Configure data cleaning rules
   - Set up aggregation preferences
   - Validate and execute data joins
   - Configure and execute data splitting
   - Train and evaluate models
   - Select best performing model
   - Run inference on new data


🛠️ Architecture

The pipeline consists of these key components:
- **MainOrchestrator**: Controls the overall pipeline flow and step progression
- **DataGatherer**: Handles file uploads and AI-powered categorization
- **DataMapper**: Manages schema mapping and validation
- **DataCleaner**: Processes and standardizes data with AI suggestions
- **DataAggregator**: Handles data aggregation logic with AI-powered method selection
- **DataJoiner**: Manages the two-phase joining process
- **DataSplitter**: Handles dataset splitting for ML
- **ModelTrainer**: Manages model training and evaluation
- **ModelSelector**: Handles AI-powered model selection
- **ModelInference**: Manages prediction generation


🔒 Security
- Secure data handling
- Input validation
- Environment variables for sensitive data
- Safe data processing operations

📝 License
MIT License

🤝 Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

📧 Contact
Email: puneet@stepfunction.ai

### Database
sudo apt-get install sqlitebrowser
sqlitebrowser orchestrator.db

