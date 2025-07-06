"""
Performance Monitoring for Intelligent Document Chunking
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import os
from datetime import datetime

class ChunkingPerformanceMonitor:
    """
    Monitor and track performance metrics for document chunking strategies.
    """
    
    def __init__(self, metrics_file="chunking_metrics.json"):
        """
        Initialize the performance monitor.
        
        Args:
            metrics_file: File to store metrics history
        """
        self.metrics_file = metrics_file
        self.metrics_history = self._load_metrics_history()
        
    def _load_metrics_history(self) -> Dict:
        """Load metrics history from file if it exists."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except:
                return {"chunking_metrics": [], "retrieval_metrics": []}
        return {"chunking_metrics": [], "retrieval_metrics": []}
    
    def _save_metrics_history(self):
        """Save metrics history to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def record_chunking_metrics(self, 
                               doc_type: str, 
                               chunk_count: int, 
                               avg_chunk_size: float,
                               size_variation: float,
                               coverage: float,
                               content_preserved: bool):
        """
        Record metrics for a chunking operation.
        
        Args:
            doc_type: Type of document chunked
            chunk_count: Number of chunks created
            avg_chunk_size: Average size of chunks
            size_variation: Difference between largest and smallest chunk
            coverage: Percentage of original document covered
            content_preserved: Whether all content was preserved
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "doc_type": doc_type,
            "chunk_count": chunk_count,
            "avg_chunk_size": avg_chunk_size,
            "size_variation": size_variation,
            "coverage": coverage,
            "content_preserved": content_preserved
        }
        
        self.metrics_history["chunking_metrics"].append(metrics)
        self._save_metrics_history()
        
    def record_retrieval_metrics(self,
                                query: str,
                                doc_type: str,
                                strategy: str,
                                relevant_results: int,
                                total_results: int,
                                response_time_ms: float):
        """
        Record metrics for a retrieval operation.
        
        Args:
            query: The search query
            doc_type: Type of document retrieved
            strategy: Chunking strategy used
            relevant_results: Number of relevant results
            total_results: Total number of results
            response_time_ms: Response time in milliseconds
        """
        precision = relevant_results / total_results if total_results > 0 else 0
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "doc_type": doc_type,
            "strategy": strategy,
            "relevant_results": relevant_results,
            "total_results": total_results,
            "precision": precision,
            "response_time_ms": response_time_ms
        }
        
        self.metrics_history["retrieval_metrics"].append(metrics)
        self._save_metrics_history()
    
    def get_chunking_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for chunking metrics.
        
        Returns:
            Dictionary of summary metrics
        """
        if not self.metrics_history["chunking_metrics"]:
            return {}
        
        df = pd.DataFrame(self.metrics_history["chunking_metrics"])
        
        summary = {
            "total_documents_processed": len(df),
            "avg_chunks_per_document": df["chunk_count"].mean(),
            "avg_chunk_size": df["avg_chunk_size"].mean(),
            "content_preservation_rate": df["content_preserved"].mean() * 100,
            "by_doc_type": {}
        }
        
        # Group by document type
        for doc_type, group in df.groupby("doc_type"):
            summary["by_doc_type"][doc_type] = {
                "count": len(group),
                "avg_chunks": group["chunk_count"].mean(),
                "avg_chunk_size": group["avg_chunk_size"].mean(),
                "avg_coverage": group["coverage"].mean() * 100
            }
            
        return summary
    
    def get_retrieval_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for retrieval metrics.
        
        Returns:
            Dictionary of summary metrics
        """
        if not self.metrics_history["retrieval_metrics"]:
            return {}
        
        df = pd.DataFrame(self.metrics_history["retrieval_metrics"])
        
        summary = {
            "total_queries": len(df),
            "avg_precision": df["precision"].mean() * 100,
            "avg_response_time_ms": df["response_time_ms"].mean(),
            "by_strategy": {},
            "by_doc_type": {}
        }
        
        # Group by strategy
        for strategy, group in df.groupby("strategy"):
            summary["by_strategy"][strategy] = {
                "count": len(group),
                "avg_precision": group["precision"].mean() * 100,
                "avg_response_time_ms": group["response_time_ms"].mean()
            }
            
        # Group by document type
        for doc_type, group in df.groupby("doc_type"):
            summary["by_doc_type"][doc_type] = {
                "count": len(group),
                "avg_precision": group["precision"].mean() * 100,
                "avg_response_time_ms": group["response_time_ms"].mean()
            }
            
        return summary
    
    def compare_chunking_strategies(self, strategies: List[str] = None) -> Dict[str, Any]:
        """
        Compare different chunking strategies based on retrieval metrics.
        
        Args:
            strategies: List of strategies to compare (if None, compare all)
            
        Returns:
            Dictionary with comparison metrics
        """
        if not self.metrics_history["retrieval_metrics"]:
            return {}
        
        df = pd.DataFrame(self.metrics_history["retrieval_metrics"])
        
        if strategies:
            df = df[df["strategy"].isin(strategies)]
            
        if df.empty:
            return {}
            
        comparison = {}
        
        for strategy, group in df.groupby("strategy"):
            comparison[strategy] = {
                "query_count": len(group),
                "avg_precision": group["precision"].mean() * 100,
                "avg_response_time_ms": group["response_time_ms"].mean(),
                "by_doc_type": {}
            }
            
            # Further break down by document type
            for doc_type, subgroup in group.groupby("doc_type"):
                comparison[strategy]["by_doc_type"][doc_type] = {
                    "query_count": len(subgroup),
                    "avg_precision": subgroup["precision"].mean() * 100,
                    "avg_response_time_ms": subgroup["response_time_ms"].mean()
                }
                
        return comparison
    
    def generate_performance_report(self) -> str:
        """
        Generate a human-readable performance report.
        
        Returns:
            Report text
        """
        chunking_summary = self.get_chunking_metrics_summary()
        retrieval_summary = self.get_retrieval_metrics_summary()
        
        if not chunking_summary or not retrieval_summary:
            return "Insufficient data to generate report."
        
        report = []
        report.append("# Intelligent Document Chunking Performance Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Chunking Performance")
        report.append(f"Total Documents Processed: {chunking_summary['total_documents_processed']}")
        report.append(f"Average Chunks per Document: {chunking_summary['avg_chunks_per_document']:.2f}")
        report.append(f"Average Chunk Size: {chunking_summary['avg_chunk_size']:.2f} characters")
        report.append(f"Content Preservation Rate: {chunking_summary['content_preservation_rate']:.2f}%")
        
        report.append("\n### By Document Type")
        for doc_type, metrics in chunking_summary["by_doc_type"].items():
            report.append(f"\n#### {doc_type.capitalize()}")
            report.append(f"Count: {metrics['count']}")
            report.append(f"Average Chunks: {metrics['avg_chunks']:.2f}")
            report.append(f"Average Chunk Size: {metrics['avg_chunk_size']:.2f} characters")
            report.append(f"Average Coverage: {metrics['avg_coverage']:.2f}%")
        
        report.append("\n## Retrieval Performance")
        report.append(f"Total Queries: {retrieval_summary['total_queries']}")
        report.append(f"Average Precision: {retrieval_summary['avg_precision']:.2f}%")
        report.append(f"Average Response Time: {retrieval_summary['avg_response_time_ms']:.2f} ms")
        
        report.append("\n### By Strategy")
        for strategy, metrics in retrieval_summary["by_strategy"].items():
            report.append(f"\n#### {strategy}")
            report.append(f"Count: {metrics['count']}")
            report.append(f"Average Precision: {metrics['avg_precision']:.2f}%")
            report.append(f"Average Response Time: {metrics['avg_response_time_ms']:.2f} ms")
        
        return "\n".join(report)
    
    def plot_strategy_comparison(self, strategies: List[str] = None) -> plt.Figure:
        """
        Generate a plot comparing different chunking strategies.
        
        Args:
            strategies: List of strategies to compare (if None, compare all)
            
        Returns:
            Matplotlib figure
        """
        comparison = self.compare_chunking_strategies(strategies)
        
        if not comparison:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient data for comparison", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Prepare data for plotting
        strategy_names = list(comparison.keys())
        precision_values = [comparison[s]["avg_precision"] for s in strategy_names]
        response_times = [comparison[s]["avg_response_time_ms"] for s in strategy_names]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Precision comparison
        ax1.bar(strategy_names, precision_values, color='skyblue')
        ax1.set_title('Retrieval Precision by Chunking Strategy')
        ax1.set_ylabel('Precision (%)')
        ax1.set_ylim(0, 100)
        
        # Response time comparison
        ax2.bar(strategy_names, response_times, color='lightgreen')
        ax2.set_title('Response Time by Chunking Strategy')
        ax2.set_ylabel('Response Time (ms)')
        
        plt.tight_layout()
        return fig

# Example usage
def main():
    # Create a monitor instance
    monitor = ChunkingPerformanceMonitor()
    
    # Record some example metrics
    # Chunking metrics
    monitor.record_chunking_metrics(
        doc_type="technical",
        chunk_count=5,
        avg_chunk_size=800,
        size_variation=200,
        coverage=0.98,
        content_preserved=True
    )
    
    monitor.record_chunking_metrics(
        doc_type="code",
        chunk_count=3,
        avg_chunk_size=1200,
        size_variation=500,
        coverage=0.99,
        content_preserved=True
    )
    
    # Retrieval metrics
    monitor.record_retrieval_metrics(
        query="How to authenticate API",
        doc_type="technical",
        strategy="adaptive",
        relevant_results=2,
        total_results=3,
        response_time_ms=150
    )
    
    monitor.record_retrieval_metrics(
        query="How to authenticate API",
        doc_type="technical",
        strategy="uniform",
        relevant_results=1,
        total_results=3,
        response_time_ms=120
    )
    
    # Generate and print report
    report = monitor.generate_performance_report()
    print(report)
    
    # Generate comparison plot
    fig = monitor.plot_strategy_comparison()
    plt.savefig("strategy_comparison.png")
    print("Plot saved as strategy_comparison.png")

if __name__ == "__main__":
    main() 