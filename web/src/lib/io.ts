// Reading an uploaded file in the browser, and CSV downloads. Nothing leaves the browser.
import readXlsxFile, { readSheetNames } from 'read-excel-file'
import Papa from 'papaparse'
import type { Row } from '../engine/core'

export class DataError extends Error {}

function toRows(table: any[][]): Row[] {
  if (!table.length) return []
  const head = table[0].map(h => String(h ?? '').trim())
  return table.slice(1).filter(r => r.some(v => v !== null && v !== '')).map(r => Object.fromEntries(head.map((h, i) => [h, r[i] ?? null])))
}

export async function readMerchants(file: File): Promise<Row[]> {
  if (!file.size) throw new DataError('The file is empty.')
  if (file.name.toLowerCase().endsWith('.csv')) {
    const text = await file.text()
    const p = Papa.parse<Row>(text, { header: true, dynamicTyping: true, skipEmptyLines: true })
    if (!p.data.length) throw new DataError('This CSV has no rows. Save it again as a CSV with one header row.')
    return p.data
  }
  let sheets: string[]
  try { sheets = await readSheetNames(file) } catch { throw new DataError('This is not a readable Excel file. Upload an .xlsx file (or a .csv) with the P2_Merchants columns.') }
  if (sheets.includes('P2_Merchants')) return toRows(await readXlsxFile(file, { sheet: 'P2_Merchants' }))
  for (const s of sheets) {
    const rows = toRows(await readXlsxFile(file, { sheet: s }))
    if (rows.length && 'merchant_id' in rows[0]) return rows
  }
  throw new DataError(`No tab called P2_Merchants, and no tab with a merchant_id column. Tabs found: ${sheets.join(', ')}.`)
}

export function downloadCsv(name: string, rows: Row[]) {
  const csv = '﻿' + Papa.unparse(rows)
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
  const a = document.createElement('a'); a.href = url; a.download = name; a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export const store = {
  get<T>(key: string, fallback: T): T { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) as T : fallback } catch { return fallback } },
  set(key: string, value: unknown) { try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* private mode: keep it in memory only */ } },
}
