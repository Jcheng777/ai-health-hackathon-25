import { pgTable, serial, integer, varchar, numeric, text, jsonb } from 'drizzle-orm/pg-core';

export const claim = pgTable('claim', {
  id: serial('id').primaryKey(),
  providerId: integer('provider_id'),
  procedureCode: varchar('procedure_code', { length: 255 }),
  diagnosisCode: varchar('diagnosis_code', { length: 255 }),
  billedAmount: numeric('billed_amount'),
  insuranceType: varchar('insurance_type', { length: 255 }),
  additionalInfo: text('additional_info'),
  prediction: varchar('prediction', { length: 255 }),
  confidenceScore: numeric('confidence_score'),
  likelihoodPercent: numeric('likelihood_percent'),
  denialReasons: jsonb('denial_reasons'),
  nextSteps: jsonb('next_steps'),
  analysisDetails: jsonb('analysis_details'),
}); 