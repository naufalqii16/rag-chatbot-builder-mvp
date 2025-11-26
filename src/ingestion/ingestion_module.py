"""
Ingestion Module for Multiple File Formats
Supports: PDF, Excel, CSV, TXT
Includes data validation and cleaning
"""

import pandas as pd
import PyPDF2
import chardet
from pathlib import Path
from typing import Union, Dict, List, Any
from datetime import datetime
import re


class DataIngestionModule:
    """
    A comprehensive data ingestion module supporting multiple file formats
    with built-in validation and cleaning capabilities.
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.xlsx', '.xls', '.csv', '.txt']
        self.ingestion_metadata = {}
    
    def ingest_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Main ingestion function that routes to specific handlers based on file type.
        
        Args:
            file_path: Path to the file to ingest
            **kwargs: Additional arguments for specific file handlers
                - sheet_name: For Excel files (default: 0)
                - encoding: For text/CSV files (default: auto-detect)
                - delimiter: For CSV files (default: ',')
                - clean_data: Whether to apply data cleaning (default: True)
        
        Returns:
            Dictionary containing:
                - 'data': Processed data (DataFrame or text)
                - 'metadata': File information and processing details
                - 'validation_report': Data quality report
                - 'status': 'success' or 'error'
                - 'message': Processing message
        """
        try:
            file_path = Path(file_path)
            
            # Validate file exists
            if not file_path.exists():
                return self._error_response(f"File not found: {file_path}")
            
            # Check file format
            file_ext = file_path.suffix.lower()
            if file_ext not in self.supported_formats:
                return self._error_response(
                    f"Unsupported format: {file_ext}. Supported: {self.supported_formats}"
                )
            
            # Route to appropriate handler
            if file_ext == '.pdf':
                result = self._ingest_pdf(file_path, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                result = self._ingest_excel(file_path, **kwargs)
            elif file_ext == '.csv':
                result = self._ingest_csv(file_path, **kwargs)
            elif file_ext == '.txt':
                result = self._ingest_txt(file_path, **kwargs)
            
            # Apply data cleaning if requested and data is DataFrame
            if kwargs.get('clean_data', True) and isinstance(result.get('data'), pd.DataFrame):
                result['data'] = self._clean_dataframe(result['data'])
                result['metadata']['cleaned'] = True
            
            # Generate validation report for DataFrames
            if isinstance(result.get('data'), pd.DataFrame):
                result['validation_report'] = self._validate_data(result['data'])
            
            result['status'] = 'success'
            return result
            
        except Exception as e:
            return self._error_response(f"Ingestion failed: {str(e)}")
    
    def _ingest_pdf(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Extract text content from PDF files."""
        text_content = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text_content.append(page.extract_text())
        
        full_text = '\n'.join(text_content)
        
        # Apply text cleaning if requested
        original_length = len(full_text)
        if kwargs.get('clean_data', True):
            full_text = self._clean_text(full_text)
        
        return {
            'data': full_text,
            'metadata': {
                'file_name': file_path.name,
                'file_type': 'PDF',
                'file_size_bytes': file_path.stat().st_size,
                'num_pages': num_pages,
                'ingestion_time': datetime.now().isoformat(),
                'character_count': len(full_text),
                'word_count': len(full_text.split()),
                'cleaned': kwargs.get('clean_data', True),
                'original_length': original_length,
                'reduction_percentage': round((1 - len(full_text)/original_length) * 100, 2) if original_length > 0 else 0
            }
        }
    
    def _ingest_excel(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Read Excel files into pandas DataFrame."""
        sheet_name = kwargs.get('sheet_name', 0)
        
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        return {
            'data': df,
            'metadata': {
                'file_name': file_path.name,
                'file_type': 'Excel',
                'file_size_bytes': file_path.stat().st_size,
                'sheet_name': sheet_name,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'ingestion_time': datetime.now().isoformat()
            }
        }
    
    def _ingest_csv(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Read CSV files into pandas DataFrame."""
        encoding = kwargs.get('encoding', self._detect_encoding(file_path))
        delimiter = kwargs.get('delimiter', ',')
        
        df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
        
        return {
            'data': df,
            'metadata': {
                'file_name': file_path.name,
                'file_type': 'CSV',
                'file_size_bytes': file_path.stat().st_size,
                'encoding': encoding,
                'delimiter': delimiter,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'ingestion_time': datetime.now().isoformat()
            }
        }
    
    def _ingest_txt(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Read text files."""
        encoding = kwargs.get('encoding', self._detect_encoding(file_path))
        
        with open(file_path, 'r', encoding=encoding) as file:
            text_content = file.read()
        
        # Apply text cleaning if requested
        original_length = len(text_content)
        if kwargs.get('clean_data', True):
            text_content = self._clean_text(text_content)
        
        return {
            'data': text_content,
            'metadata': {
                'file_name': file_path.name,
                'file_type': 'TXT',
                'file_size_bytes': file_path.stat().st_size,
                'encoding': encoding,
                'ingestion_time': datetime.now().isoformat(),
                'character_count': len(text_content),
                'line_count': len(text_content.split('\n')),
                'word_count': len(text_content.split()),
                'cleaned': kwargs.get('clean_data', True),
                'original_length': original_length,
                'reduction_percentage': round((1 - len(text_content)/original_length) * 100, 2) if original_length > 0 else 0
            }
        }
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Auto-detect file encoding."""
        with open(file_path, 'rb') as file:
            raw_data = file.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive data cleaning to DataFrame.
        
        Cleaning steps:
        1. Remove duplicate rows
        2. Strip whitespace from string columns
        3. Standardize column names (lowercase, replace spaces with underscores)
        4. Standardize null values (N/A, null, None, -, empty → NaN)
        5. Convert boolean values (Y/N, Yes/No, True/False → boolean)
        6. Remove empty rows/columns
        """
        df_cleaned = df.copy()
        
        # Standardize column names
        df_cleaned.columns = [
            re.sub(r'\s+', '_', str(col).strip().lower()) 
            for col in df_cleaned.columns
        ]
        
        # Remove duplicate rows
        df_cleaned = df_cleaned.drop_duplicates()
        
        # Strip whitespace from string columns first
        string_columns = df_cleaned.select_dtypes(include=['object']).columns
        for col in string_columns:
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
        
        # Standardize null values
        df_cleaned = self._standardize_null_values(df_cleaned)
        
        # Convert boolean values
        df_cleaned = self._convert_boolean_columns(df_cleaned)
        
        # Remove completely empty rows and columns
        df_cleaned = df_cleaned.dropna(how='all', axis=0)  # Remove empty rows
        df_cleaned = df_cleaned.dropna(how='all', axis=1)  # Remove empty columns
        
        return df_cleaned
    
    def _standardize_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize various null value representations to NaN.
        
        Converts: "N/A", "n/a", "NA", "null", "NULL", "None", "NONE", 
                  "-", "--", "---", "", " ", "nan", "NaN", "#N/A", "#NA"
        """
        null_representations = [
            'N/A', 'n/a', 'NA', 'na', 'N.A.', 'n.a.',
            'null', 'NULL', 'Null',
            'None', 'NONE', 'none',
            'nil', 'NIL', 'Nil',
            '-', '--', '---',
            '', ' ', '  ',
            'nan', 'NaN', 'NAN',
            '#N/A', '#NA', '#N/A N/A',
            'missing', 'MISSING', 'Missing',
            'undefined', 'UNDEFINED', 'Undefined',
            '?', '??',
            'blank', 'BLANK', 'Blank'
        ]
        
        df_cleaned = df.copy()
        
        # Replace null representations with NaN
        for col in df_cleaned.columns:
            # For object/string columns
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].replace(null_representations, pd.NA)
                # Also handle whitespace-only strings
                df_cleaned[col] = df_cleaned[col].apply(
                    lambda x: pd.NA if isinstance(x, str) and not x.strip() else x
                )
        
        return df_cleaned
    
    def _convert_boolean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-detect and convert boolean-like columns to actual boolean type.
        
        Converts:
        - "Y", "N" → True, False
        - "Yes", "No" → True, False
        - "True", "False" → True, False
        - "T", "F" → True, False
        - "1", "0" → True, False (if column looks boolean)
        - "ON", "OFF" → True, False
        """
        boolean_mappings = {
            # Yes/No variants
            'y': True, 'n': False,
            'yes': True, 'no': False,
            'yeah': True, 'nope': False,
            
            # True/False variants
            'true': True, 'false': False,
            't': True, 'f': False,
            
            # 1/0 as strings
            '1': True, '0': False,
        }
        
        df_cleaned = df.copy()
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                # Get unique non-null values (lowercase)
                unique_vals = df_cleaned[col].dropna().astype(str).str.lower().unique()
                
                # Check if all unique values are in boolean mappings
                if len(unique_vals) > 0 and all(val in boolean_mappings for val in unique_vals):
                    # Convert to boolean
                    df_cleaned[col] = df_cleaned[col].astype(str).str.lower().map(boolean_mappings)
                    
        return df_cleaned
    
    def _clean_text(self, text: str) -> str:
        """
        Apply comprehensive text cleaning for PDF and TXT files.
        
        Cleaning steps:
        1. Fix encoding issues
        2. Remove/normalize special characters
        3. Fix hyphenation at line breaks
        4. Remove excessive whitespace and tabs
        5. Remove multiple newlines
        6. Remove page numbers
        7. Strip leading/trailing whitespace
        8. Remove empty lines
        9. Normalize spacing
        """
        if not text:
            return text

        # 1. Fix common encoding issues
        encoding_fixes = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '–',
            'Â': '',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            '\u200b': '',  # Zero-width space
            '\u00a0': ' ',  # Non-breaking space
            '\ufeff': '',   # Byte order mark
        }
        for wrong, correct in encoding_fixes.items():
            text = text.replace(wrong, correct)

        # 2. Fix hyphenation at line breaks (e.g., "informa-\ntion" -> "information")
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

        # 3. Remove page numbers
        text = re.sub(r'^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\[\s*\d+\s*\]\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*/\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone numbers

        # 4. Remove tabs and convert to spaces
        text = text.replace('\t', ' ')

        # 5. Remove weird whitespace characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)

        # 6. Normalize multiple spaces to single space (preserve line breaks)
        text = re.sub(r'[ \f\v]+', ' ', text)

        # 7. Strip spaces at start and end of each line
        text = '\n'.join(line.strip() for line in text.split('\n'))

        # 8. Remove multiple consecutive newlines (>2)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 9. Remove empty lines at start and end
        text = text.strip()

        # 10. Remove lines with only special characters
        text = re.sub(r'^\s*[.\-_=*]{3,}\s*$', '', text, flags=re.MULTILINE)

        # 11. Normalize excessive punctuation
        text = re.sub(r'\.{2,}', '.', text)  # Multiple dots -> single dot
        text = re.sub(r',{2,}', ',', text)
        text = re.sub(r';{2,}', ';', text)
        text = re.sub(r':{2,}', ':', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'\.\.\.+', '...', text)  # Normalize ellipsis

        # 12. Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation

        # 13. Final cleanup: remove empty lines
        lines = [line for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        return text

    
    def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for DataFrame.
        
        Returns:
            Dictionary with validation metrics and quality indicators
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        
        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_cells': total_cells,
            'missing_cells': int(missing_cells),
            'missing_percentage': round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0,
            'duplicate_rows': int(df.duplicated().sum()),
            'column_info': {},
            'data_types': df.dtypes.astype(str).to_dict(),
            'boolean_columns': [],
            'null_standardization_applied': True
        }
        
        # Per-column validation
        for col in df.columns:
            col_info = {
                'missing_count': int(df[col].isna().sum()),
                'missing_percentage': round((df[col].isna().sum() / len(df) * 100), 2),
                'unique_values': int(df[col].nunique()),
                'data_type': str(df[col].dtype)
            }
            
            # Add boolean flag if column is boolean type
            if df[col].dtype == 'bool':
                col_info['is_boolean'] = True
                validation_report['boolean_columns'].append(col)
                col_info['true_count'] = int(df[col].sum())
                col_info['false_count'] = int((~df[col]).sum())
            
            validation_report['column_info'][col] = col_info
        
        return validation_report
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Generate standardized error response."""
        return {
            'status': 'error',
            'message': message,
            'data': None,
            'metadata': {},
            'validation_report': {}
        }


# Convenience function for quick ingestion
def ingest(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    Quick ingestion function.
    
    Usage:
        result = ingest('data.csv')
        result = ingest('report.pdf')
        result = ingest('spreadsheet.xlsx', sheet_name='Sheet1')
    """
    module = DataIngestionModule()
    return module.ingest_file(file_path, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example 1: Ingest CSV file
    # result = ingest('data.csv')
    # print(f"Status: {result['status']}")
    # print(f"Rows: {result['metadata'].get('rows')}")
    # print(f"Validation: {result['validation_report']}")
    
    # Example 2: Ingest Excel with specific sheet
    # result = ingest('workbook.xlsx', sheet_name='Sales')
    
    # Example 3: Ingest PDF
    # result = ingest('document.pdf')
    # print(f"Text length: {result['metadata'].get('character_count')}")
    
    print("Ingestion module ready. Use ingest() function to process files.")


def process_and_index_files(file_paths: List[Path]) -> Dict[str, Any]:
    """
    Process uploaded files: chunk and index to vector database.
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        Dictionary with processing results
    """
    try:
        import sys
        from pathlib import Path as PathLib
        sys.path.insert(0, str(PathLib(__file__).parent.parent))
        
        from config.settings import settings
        from vectorstore.index_builder import IndexBuilder
        from ingestion.user_upload_chunking import process_user_files
        
        # Step 1: Ingest and chunk files
        all_chunks = []
        
        for file_path in file_paths:
            # Ingest file
            module = DataIngestionModule()
            result = module.ingest_file(str(file_path))
            
            if result['status'] != 'success':
                continue
            
            # Convert to text
            if isinstance(result['data'], pd.DataFrame):
                # For tabular data, convert to readable text
                text_content = result['data'].to_string(index=False)
            else:
                # For text/PDF
                text_content = result['data']
            
            # Chunk the text
            chunks = process_user_files(
                text_content=text_content,
                source_name=file_path.name,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return {
                'success': False,
                'error': 'No chunks generated from files'
            }
        
        # Step 2: Build index (embed and store in Qdrant)
        builder = IndexBuilder(collection_name=settings.QDRANT_USER_UPLOAD_COLLECTION)
        success = builder.build_index(
            chunks=all_chunks,
            batch_size=20,
            create_new_collection=False  # Append to existing collection
        )
        
        if success:
            return {
                'success': True,
                'total_chunks': len(all_chunks),
                'vectors_indexed': len(all_chunks),
                'files_processed': len(file_paths)
            }
        else:
            return {
                'success': False,
                'error': 'Failed to index chunks'
            }
            
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

