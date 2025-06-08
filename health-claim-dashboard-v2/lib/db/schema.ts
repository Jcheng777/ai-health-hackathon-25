import { pgTable, serial, integer, varchar, numeric, text } from 'drizzle-orm/pg-core';

export const claim = pgTable('claim', {
  id: serial('id').primaryKey(),
  providerId: integer('provider_id'),
  procedureCode: varchar('procedure_code', { length: 255 }),
  diagnosisCode: varchar('diagnosis_code', { length: 255 }),
  billedAmount: numeric('billed_amount'),
  insuranceType: varchar('insurance_type', { length: 255 }),
  additionalInfo: text('additional_info'),
}); 