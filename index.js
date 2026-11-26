import * as dotenv from 'dotenv';
dotenv.config();
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Retry helper for rate limits (429) and temporary errors
async function withRetry(fn, tries = 4) {
  for (let attempt = 1; attempt <= tries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt === tries) throw err;
      console.log(`retry ${attempt}/${tries - 1} after error: ${err.message?.slice(0, 80)}`);
      await sleep(1500 * attempt);
    }
  }
}

async function indexpdf() {
  // 1. Load and split the PDF
  const pdfLoader = new PDFLoader('./dsa.pdf');
  const rawDocs = await pdfLoader.load();
  console.log('PDF loaded, pages:', rawDocs.length);

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

  // 2. Remove empty or tiny chunks (these caused the "dimension 0" error)
  const docs = chunkedDocs.filter(
    (d) => d.pageContent && d.pageContent.trim().length > 20
  );
  console.log(`chunks: ${chunkedDocs.length}, usable: ${docs.length}`);

  // 3. Embeddings (3072 dimensions, matching the dsa-rag index)
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'gemini-embedding-001',
  });

  // 4. Connect to Pinecone and check the dimensions match
  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  const test = await embeddings.embedQuery('hello world');
  const stats = await pineconeIndex.describeIndexStats();
  console.log('embedding length:', test.length, '| index dimension:', stats.dimension);
  if (test.length !== stats.dimension) {
    throw new Error(
      `Dimension mismatch: embeddings are ${test.length}, index is ${stats.dimension}`
    );
  }

  // 5. Clear old partial data so you don't get duplicates
  try {
    await pineconeIndex.deleteAll();
    console.log('old vectors cleared');
  } catch (e) {
    console.log('nothing to clear');
  }

  // 6. Embed each chunk and upsert in batches of 50
  let batch = [];
  let stored = 0;

  for (let i = 0; i < docs.length; i++) {
    let values;
    try {
      values = await withRetry(() => embeddings.embedQuery(docs[i].pageContent));
    } catch (e) {
      console.log(`skipping chunk ${i}: ${e.message?.slice(0, 80)}`);
      continue;
    }

    if (!values || values.length !== stats.dimension) {
      console.log(`skipping chunk ${i}, bad vector length:`, values?.length);
      continue;
    }

    batch.push({
      id: `dsa-${i}`,
      values,
      metadata: {
        text: docs[i].pageContent,
        page: docs[i].metadata?.loc?.pageNumber ?? 0,
      },
    });

    if (batch.length === 50 || i === docs.length - 1) {
      if (batch.length) {
        await withRetry(() => pineconeIndex.upsert(batch));
        stored += batch.length;
        console.log(`upserted ${stored}/${docs.length}`);
      }
      batch = [];
    }

    await sleep(100); // gentle on the rate limit
  }

  console.log(`Data stored: ${stored} vectors`);
}

indexpdf().catch((err) => {
  console.error('Indexing failed:', err);
  process.exit(1);
});