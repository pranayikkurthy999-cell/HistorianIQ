from fpdf import FPDF
import datetime

class PDFReportGenerator(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'HistorianIQ: Data Conditioning Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")} | Page {self.page_no()}', 0, 0, 'C')

    def sanitize_text(self, text):
        """Replaces UI emojis with PDF-safe ASCII text."""
        clean_text = str(text).replace('⚠', '[WARNING]').replace('✔', '[OK]')
        # Catch any other weird characters and replace them with a '?'
        return clean_text.encode('latin-1', 'replace').decode('latin-1')

    def generate_report(self, tag_name, quality_metrics, filter_used, snr_before, snr_after, ml_score, recommendations):
        self.add_page()
        
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, f'Tag Analyzed: {tag_name}', 0, 1)
        
        self.set_font('Arial', '', 11)
        self.cell(0, 8, f'Original Quality Score: {quality_metrics["quality_score"]:.1f} / 100', 0, 1)
        self.cell(0, 8, f'Missing Data: {quality_metrics["missing_pct"]:.2f}% ({quality_metrics["missing_samples"]} samples)', 0, 1)
        self.cell(0, 8, f'Frozen Sensor: {quality_metrics["frozen_pct"]:.2f}%', 0, 1)
        self.ln(5)
        
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Conditioning Applied', 0, 1)
        self.set_font('Arial', '', 11)
        self.cell(0, 8, f'Filter Strategy: {filter_used}', 0, 1)
        self.cell(0, 8, f'Initial SNR: {snr_before:.2f} dB', 0, 1)
        self.cell(0, 8, f'Final SNR (After Filtering): {snr_after:.2f} dB', 0, 1)
        self.ln(5)
        
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Machine Learning Readiness', 0, 1)
        self.set_font('Arial', '', 11)
        self.cell(0, 8, f'Readiness Score: {ml_score}%', 0, 1)
        
        # Sanitize each recommendation before writing it to the PDF
        for rec in recommendations:
            clean_rec = self.sanitize_text(rec)
            self.cell(0, 8, clean_rec, 0, 1)
            
        return self.output(dest='S').encode('latin-1')