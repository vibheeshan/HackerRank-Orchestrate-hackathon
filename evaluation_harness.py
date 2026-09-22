#!/usr/bin/env python3
"""
Evaluation Harness - Validate output against sample requests
Compares generated output with expected output from sample_requests.csv
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Tracks evaluation metrics"""
    
    def __init__(self):
        self.total_requests = 0
        self.exact_matches = 0
        self.partial_matches = 0
        self.mismatches = []
        
        self.amount_errors = []
        self.status_errors = []
        self.method_errors = []
    
    def evaluate_field(self, field_name: str, expected: str, actual: str) -> bool:
        """Compare field values"""
        
        if field_name == 'amount_safe_to_pay':
            try:
                expected_float = float(expected)
                actual_float = float(actual)
                error = abs(expected_float - actual_float)
                
                if error <= 1.0:  # Allow $1 variance
                    return True
                else:
                    self.amount_errors.append({
                        'expected': expected_float,
                        'actual': actual_float,
                        'error': error
                    })
                    return False
            except ValueError:
                return expected == actual
        
        elif field_name in ['affordability_status', 'recommended_payment_method']:
            match = expected.lower().strip() == actual.lower().strip()
            if not match:
                if field_name == 'affordability_status':
                    self.status_errors.append((expected, actual))
                else:
                    self.method_errors.append((expected, actual))
            return match
        
        else:
            # String comparison
            return expected.lower().strip() == actual.lower().strip()
    
    def report(self) -> str:
        """Generate evaluation report"""
        
        accuracy = (self.exact_matches / max(1, self.total_requests)) * 100
        
        report = f"""
{'='*80}
EVALUATION REPORT
{'='*80}

Total Requests Evaluated: {self.total_requests}
Exact Matches: {self.exact_matches}
Partial Matches: {self.partial_matches}
Accuracy: {accuracy:.2f}%

Amount Errors: {len(self.amount_errors)}
- Average error: ${sum(e['error'] for e in self.amount_errors) / max(1, len(self.amount_errors)):.2f}
- Max error: ${max((e['error'] for e in self.amount_errors), default=0):.2f}

Status Mismatches: {len(self.status_errors)}
{self._format_errors(self.status_errors)}

Method Mismatches: {len(self.method_errors)}
{self._format_errors(self.method_errors)}

Total Mismatches: {len(self.mismatches)}

{'='*80}
"""
        return report
    
    @staticmethod
    def _format_errors(errors: List[Tuple], limit: int = 5) -> str:
        """Format error list"""
        if not errors:
            return ""
        
        formatted = ""
        for i, (expected, actual) in enumerate(errors[:limit]):
            formatted += f"  - Expected: '{expected}' | Actual: '{actual}'\n"
        
        if len(errors) > limit:
            formatted += f"  ... and {len(errors) - limit} more\n"
        
        return formatted


class Evaluator:
    """Evaluate predictions against expected outputs"""
    
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
    
    def load_sample_requests(self) -> List[Dict]:
        """Load sample requests with expected outputs"""
        file_path = self.dataset_dir / 'sample_requests.csv'
        
        if not file_path.exists():
            logger.warning(f"Sample requests file not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    
    def load_generated_output(self, output_path: str = "output.csv") -> Dict[str, Dict]:
        """Load generated output file"""
        output = {}
        
        path = Path(output_path)
        if not path.exists():
            logger.error(f"Output file not found: {output_path}")
            return output
        
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                request_id = row.get('request_id')
                output[request_id] = row
        
        return output
    
    def evaluate(self, output_path: str = "output.csv") -> EvaluationMetrics:
        """
        Evaluate generated output against sample requests.
        
        Args:
            output_path: Path to generated output.csv
        
        Returns:
            EvaluationMetrics object
        """
        
        logger.info("Starting evaluation...")
        
        sample_requests = self.load_sample_requests()
        generated_output = self.load_generated_output(output_path)
        
        metrics = EvaluationMetrics()
        
        for expected in sample_requests:
            request_id = expected.get('request_id')
            actual = generated_output.get(request_id, {})
            
            metrics.total_requests += 1
            
            # Compare each field
            fields_to_check = [
                'amount_safe_to_pay',
                'affordability_status',
                'recommended_payment_method',
                'payment_plan',
                'earliest_date_for_full_payment',
                'spending_changes_needed',
                'decision_explanation'
            ]
            
            field_matches = 0
            for field in fields_to_check:
                expected_val = expected.get(field, '')
                actual_val = actual.get(field, '')
                
                if metrics.evaluate_field(field, expected_val, actual_val):
                    field_matches += 1
            
            if field_matches == len(fields_to_check):
                metrics.exact_matches += 1
            elif field_matches > len(fields_to_check) / 2:
                metrics.partial_matches += 1
            else:
                metrics.mismatches.append({
                    'request_id': request_id,
                    'fields_matched': field_matches,
                    'total_fields': len(fields_to_check)
                })
        
        return metrics


def main():
    """Run evaluation against sample_requests.csv"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    evaluator = Evaluator()
    sample_requests = evaluator.load_sample_requests()
    sample_ids = {r.get('request_id') for r in sample_requests if r.get('request_id')}
    
    generated_output = evaluator.load_generated_output()
    
    # If output.csv doesn't contain sample request IDs, run pipeline on sample_requests.csv for evaluation
    if sample_ids and not any(sid in generated_output for sid in sample_ids):
        logger.info("Generating predictions for sample_requests.csv for evaluation...")
        import importlib.util
        root_dir = Path(__file__).resolve().parent
        main_path = root_dir / "code" / "main.py"
        spec = importlib.util.spec_from_file_location("agent_main", str(main_path))
        agent_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_main)
        
        loader = agent_main.DataLoader()
        calculator = agent_main.FinancialCalculator(loader)
        engine = agent_main.DecisionEngine(loader, calculator)
        
        sample_decisions = []
        for row in loader.sample_requests:
            req = agent_main.FinancialRequest.from_csv_row(row)
            dec = engine.decide(req)
            sample_decisions.append(dec)
            
        # Write temporary sample output for evaluation
        sample_out_path = Path("sample_output.csv")
        fieldnames = [
            'request_id', 'amount_safe_to_pay', 'affordability_status',
            'recommended_payment_method', 'payment_plan', 'earliest_date_for_full_payment',
            'spending_changes_needed', 'decision_explanation'
        ]
        with open(sample_out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in sample_decisions:
                writer.writerow(d.to_csv_row())
                
        metrics = evaluator.evaluate("sample_output.csv")
    else:
        metrics = evaluator.evaluate("output.csv")
    
    report_text = metrics.report()
    print(report_text)
    
    report_path = Path("code/evaluation/evaluation_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Report saved to {report_path}")


if __name__ == '__main__':
    main()
