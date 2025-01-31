from sfn_blueprint import Task
import pandas as pd
from utils.model_manager import ModelManager


class Step9ModelInferenceOrchestrator:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.model_manager = ModelManager()

    def orchestrate(self):
        # Skip if not at step 9
        if self.session.get('current_step', 1) != 9:
            if self.session.get('current_step', 1) > 9:
                self._display_inference_summary()
            return

        if not self.session.get('inference_complete'):
            self._run_inference()
        else:
            self._show_download_options()

    def _run_inference(self):
        self.view.display_header("Step 9: Model Inference")

        # Get selected model and inference data
        model_id = self.session.get('selected_model_id')
        infer_df = self.session.get('infer_df')
        print(f">>>model_id: {model_id}")  # temp
        print(f">>>infer_df: {infer_df}")  # temp
        if not model_id or infer_df is None:
            self.view.show_message(
                "❌ Required data not found. Please complete previous steps.", "error")
            return

        try:
            # Load model and metadata
            with self.view.display_spinner("Loading selected model..."):
                model, metadata = self.model_manager.load_model(model_id)

                # Get training features from metadata
                training_features = metadata.get('training_features', [])
                if not training_features:
                    raise ValueError(
                        "Training features information not found in model metadata")

                # Select only the features used in training
                infer_df = infer_df[training_features]

            # Run inference
            with self.view.display_spinner("Running inference..."):
                predictions = model.predict(infer_df)
                prediction_probs = model.predict_proba(infer_df)[:, 1]

            # Add predictions to dataframe
            result_df = infer_df.copy()
            result_df['prediction'] = predictions
            result_df['prediction_probability'] = prediction_probs

            # Store results
            self.session.set('inference_results', result_df)
            self.session.set('inference_complete', True)

            # Show results preview
            self.view.display_subheader("Inference Results Preview")
            self.view.display_dataframe(result_df.head())

            self._show_download_options()

        except Exception as e:
            self.view.show_message(
                f"❌ Error during inference: {str(e)}", "error")

    def _show_download_options(self):
        self.view.display_markdown("---")
        self.view.display_subheader("Download Options")

        col1, col2 = self.view.create_columns(2)

        with col1:
            if self.view.display_button("📥 Download Results (CSV)"):
                result_df = self.session.get('inference_results')
                if result_df is not None:
                    self.view.download_dataframe(
                        result_df,
                        "inference_results.csv",
                        "Download Results"
                    )

        with col2:
            if self.view.display_button("🏁 Finish Pipeline"):
                try:
                    # Cleanup models
                    if self.model_manager.cleanup():
                        self.view.show_message(
                            "✅ Models and registry cleaned up successfully", "success")
                    else:
                        self.view.show_message(
                            "⚠️ Error cleaning up models", "warning")
                        return

                    # Reset session only after successful cleanup
                    self.session.clear()
                    self.session.set('pipeline_complete', True)
                    self.session.set('current_step', 10)
                    self.view.rerun_script()
                except Exception as e:
                    self.view.show_message(
                        f"❌ Error during pipeline completion: {str(e)}", "error")

    def _display_inference_summary(self):
        self.view.display_header("Step 9: Model Inference ✅")
        if self.session.get('inference_complete'):
            self.view.show_message(
                "Inference completed successfully!", "success")
