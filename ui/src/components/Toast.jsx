export default function Toast({ kind, msg }) {
  return <div className={`toast ${kind}`}>{msg}</div>
}
