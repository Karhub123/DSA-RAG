import * as dotenv from 'dotenv';
dotenv.config();
import readline from 'node:readline/promises';
import { GoogleGenAI } from '@google/genai';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
const MODEL = 'gemini-3.5-flash'; // swap for a current model your key can use
const MIN_SCORE = 0.5;            // tune this by testing real questions
const MAX_TURNS = 6;
const ai = new GoogleGenAI({});
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: 'gemini-embedding-001', // same model as indexing
});
const index = new Pinecone().Index(process.env.PINECONE_INDEX_NAME);

const History = [];

async function rewriteQuery(question) {
  if (History.length === 0) return question;
  const res = await ai.models.generateContent({
    model: MODEL,
    contents: [...History, { role: 'user', parts: [{ text: question }] }],
    config: {
      systemInstruction: `Rewrite the user's latest question as a complete, standalone question
that makes sense without the chat history. Resolve words like "it", "that", "its".
Output ONLY the rewritten question.`,
    },
  });
  return res.text.trim();
}

async function chat(question) {
  // 1. Enhance the query
  const standalone = await rewriteQuery(question);

  // 2. Embed and search
  const queryVector = await embeddings.embedQuery(standalone);
  const { matches } = await index.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
  });
  // 3. Keep only relevant chunks
  const good = matches.filter((m) => m.score >= MIN_SCORE && m.metadata?.text);
  if (good.length === 0) {
    return 'I could not find the answer in the provided document.';
  }
  const context = good.map((m) => m.metadata.text).join('\n\n---\n\n');
  // 4. Answer from context only
  History.push({ role: 'user', parts: [{ text: question }] });
  const res = await ai.models.generateContent({
    model: MODEL,
    contents: History,
    config: {
      systemInstruction: `You are a Data Structures and Algorithms expert.
Answer ONLY from the context below. If the answer is not there, say
"I could not find the answer in the provided document."
Keep answers clear, concise and educational.

Context:
${context}`,
    },
  });

  History.push({ role: 'model', parts: [{ text: res.text }] });
  while (History.length > MAX_TURNS * 2) History.splice(0, 2); // trim old turns
  return res.text;
}

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
while (true) {
  const q = await rl.question('Ask: ');
  if (q.trim().toLowerCase() === 'exit') break;
  try {
    console.log('\n' + (await chat(q)) + '\n');
  } catch (e) {
    console.error('Error:', e.message);
  }
}
rl.close();