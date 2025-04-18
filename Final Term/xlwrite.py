import xlwt
from datetime import datetime
from xlrd import open_workbook
from xlutils.copy import copy
from pathlib import Path


def output(filename, sheet, num, name, present):
    # Ensure the directory exists
    Path("attendance_sheet").mkdir(exist_ok=True)

    # Define the file path
    file_path = Path(f"attendance_sheet/{filename}{datetime.now().date()}.xls")

    if file_path.is_file():
        # Open the existing file
        rb = open_workbook(file_path)
        book = copy(rb)
        sh = book.get_sheet(0)

        # Check if the date is already written in the first cell
        try:
            first_cell_value = rb.sheet_by_index(0).cell_value(0, 0)
        except IndexError:
            first_cell_value = None
    else:
        # Create a new workbook and sheet
        book = xlwt.Workbook()
        sh = book.add_sheet(sheet)
        first_cell_value = None

        # Add headers for new sheets
        style_header = xlwt.easyxf('font: name Times New Roman, color-index red, bold on')
        sh.write(1, 0, "Name", style_header)
        sh.write(1, 1, "Present", style_header)

    # Write the date if not already written
    if not first_cell_value:
        style_date = xlwt.easyxf(num_format_str='D-MMM-YY')
        sh.write(0, 0, datetime.now().date(), style_date)

    # Check for duplicates in the attendance list
    if file_path.is_file():
        rb_sheet = rb.sheet_by_index(0)
        existing_names = [rb_sheet.cell_value(row, 0) for row in range(2, rb_sheet.nrows)]
    else:
        existing_names = []

    if name not in existing_names:
        # Write attendance data
        sh.write(num + 1, 0, name)
        sh.write(num + 1, 1, present)

    # Save the workbook
    fullname = file_path.as_posix()
    book.save(fullname)
    return fullname
