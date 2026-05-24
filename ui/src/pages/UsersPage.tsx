import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { UserCheck, UserX } from 'lucide-react'
import { approveUser, listUsers, revokeUser } from '@/api/admin'
import { Button } from '@/components/ui/button'
import type { UserContext } from '@/types'

interface UserTableProps {
  kind: 'pending' | 'approved'
  users: UserContext[]
  action: (user: UserContext) => React.ReactNode
}

function UserTable({ kind, users, action }: UserTableProps) {
  return (
    <div className="bg-white border border-[#d0d0c8] rounded-[4px] shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden">
      <table className="ed-table">
        <thead>
          <tr>
            <th>Email</th>
            <th>User ID</th>
            <th style={{ width: '14rem' }}></th>
          </tr>
        </thead>
        <tbody>
          {users.length === 0 ? (
            <tr>
              <td colSpan={3} className="text-center py-8">
                <span className="font-['DM_Serif_Display'] italic text-[#888888]">
                  {kind === 'pending' ? 'No users awaiting approval' : 'No approved users'}
                </span>
              </td>
            </tr>
          ) : (
            users.map(user => (
              <tr key={user.user_id}>
                <td className="font-['Outfit'] text-[0.9rem] text-[#1a1a1a]">
                  {user.email ?? '—'}
                </td>
                <td className="font-['IBM_Plex_Mono'] text-[0.74rem] text-[#888888]">
                  {user.user_id}
                </td>
                <td className="text-right">{action(user)}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

export function UsersPage() {
  const queryClient = useQueryClient()

  const { data: pending = [], isLoading: loadingPending } = useQuery({
    queryKey: ['users', 'pending'],
    queryFn: () => listUsers('pending'),
  })

  const { data: approved = [], isLoading: loadingApproved } = useQuery({
    queryKey: ['users', 'approved'],
    queryFn: () => listUsers('approved'),
  })

  const approveMutation = useMutation({
    mutationFn: (user: UserContext) => approveUser(user.user_id, user.email),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', 'pending'] })
      queryClient.invalidateQueries({ queryKey: ['users', 'approved'] })
    },
  })

  const revokeMutation = useMutation({
    mutationFn: (user_id: string) => revokeUser(user_id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', 'pending'] })
      queryClient.invalidateQueries({ queryKey: ['users', 'approved'] })
    },
  })

  return (
    <div className="ed-page">
      <header className="mb-10">
        <div className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888] mb-3">
          Admin · Membership
        </div>
        <h1 className="font-['DM_Serif_Display'] text-[2.4rem] leading-[1.05] text-[#1a1a1a] mb-3">
          Users
        </h1>
        <p className="font-['Outfit'] text-[1rem] text-[#3a3a36] max-w-2xl">
          Approve new signups and manage access to the training dashboard.
        </p>
        <hr className="ed-rule mt-7 mb-0" />
      </header>

      <section className="mb-12">
        <div className="flex items-baseline gap-3 mb-4">
          <span className="ed-step-circle ed-step-circle--active text-[0.7rem]">01</span>
          <h2 className="font-['DM_Serif_Display'] text-[1.4rem] text-[#1a1a1a]">
            Awaiting approval
          </h2>
          <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#888888] ml-auto">
            {pending.length} {pending.length === 1 ? 'request' : 'requests'}
          </span>
        </div>
        {loadingPending ? (
          <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888]">
            Loading
          </span>
        ) : (
          <UserTable
            kind="pending"
            users={pending}
            action={(user) => (
              <Button
                size="sm"
                onClick={() => approveMutation.mutate(user)}
                disabled={approveMutation.isPending}
                aria-label={`Approve ${user.email ?? user.user_id}`}
              >
                <UserCheck className="h-3 w-3" />Approve
              </Button>
            )}
          />
        )}
      </section>

      <section>
        <div className="flex items-baseline gap-3 mb-4">
          <span className="ed-step-circle text-[0.7rem]">02</span>
          <h2 className="font-['DM_Serif_Display'] text-[1.4rem] text-[#1a1a1a]">
            Approved users
          </h2>
          <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.14em] text-[#888888] ml-auto">
            {approved.length} {approved.length === 1 ? 'member' : 'members'}
          </span>
        </div>
        {loadingApproved ? (
          <span className="font-['IBM_Plex_Mono'] text-[0.7rem] uppercase tracking-[0.18em] text-[#888888]">
            Loading
          </span>
        ) : (
          <UserTable
            kind="approved"
            users={approved}
            action={(user) => (
              <Button
                size="sm"
                variant="destructive"
                onClick={() => revokeMutation.mutate(user.user_id)}
                disabled={
                  revokeMutation.isPending &&
                  revokeMutation.variables === user.user_id
                }
                aria-label={`Revoke ${user.email ?? user.user_id}`}
              >
                <UserX className="h-3 w-3" />Revoke
              </Button>
            )}
          />
        )}
      </section>
    </div>
  )
}
